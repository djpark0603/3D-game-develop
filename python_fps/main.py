from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pyglet

if os.getenv("PYGLET_HEADLESS") == "1":
    pyglet.options["headless"] = True

from pyglet.gl import (
    Config,
    GL_DEPTH_TEST,
    GL_TRIANGLES,
    glClearColor,
    glDisable,
    glEnable,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MOUSE_SENSITIVITY = 0.0025
MOVE_SPEED = 8.0
SPRINT_SPEED = 14.0
VERTICAL_SPEED = 7.0

VERTEX_SHADER = """
#version 150

in vec3 position;
in vec3 normal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 v_world_pos;
out vec3 v_normal;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * world_pos;
}
"""

FRAGMENT_SHADER = """
#version 150

in vec3 v_world_pos;
in vec3 v_normal;

uniform vec4 tint;
uniform vec3 light_dir;
uniform float grid_strength;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    float light = max(dot(normal, normalize(-light_dir)), 0.0);
    vec3 base = tint.rgb;

    if (grid_strength > 0.0) {
        vec2 scaled = v_world_pos.xz * grid_strength;
        vec2 grid = abs(fract(scaled - 0.5) - 0.5) / fwidth(scaled);
        float line = 1.0 - clamp(min(grid.x, grid.y), 0.0, 1.0);
        base = mix(base, base * 0.32, line);
    }

    vec3 lit = base * (0.35 + light * 0.65);
    frag_color = vec4(lit, tint.a);
}
"""


@dataclass(frozen=True)
class SceneObject:
    mesh: object
    position: Vec3
    scale: Vec3
    tint: tuple[float, float, float, float]
    rotation_y: float = 0.0
    grid_strength: float = 0.0

    def model_matrix(self) -> Mat4:
        matrix = Mat4().translate(self.position)
        if self.rotation_y:
            matrix = matrix.rotate(self.rotation_y, Vec3(0.0, 1.0, 0.0))
        return matrix.scale(self.scale)


def build_cube_mesh(program: ShaderProgram):
    faces = [
        ((0.0, 0.0, 1.0), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0.0, 0.0, -1.0), [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]),
        ((-1.0, 0.0, 0.0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
        ((1.0, 0.0, 0.0), [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
        ((0.0, 1.0, 0.0), [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
        ((0.0, -1.0, 0.0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
    ]
    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []

    for face_index, (normal, vertices) in enumerate(faces):
        base = face_index * 4
        for vertex in vertices:
            positions.extend(vertex)
            normals.extend(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    return program.vertex_list_indexed(
        24,
        GL_TRIANGLES,
        indices,
        position=("f", positions),
        normal=("f", normals),
    )


def build_plane_mesh(program: ShaderProgram):
    positions = [
        -0.5,
        0.0,
        -0.5,
        0.5,
        0.0,
        -0.5,
        0.5,
        0.0,
        0.5,
        -0.5,
        0.0,
        0.5,
    ]
    normals = [
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return program.vertex_list_indexed(
        4,
        GL_TRIANGLES,
        indices,
        position=("f", positions),
        normal=("f", normals),
    )


class FpsSandboxWindow(pyglet.window.Window):
    def __init__(self) -> None:
        visible = not pyglet.options["headless"]
        try:
            config = Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
            super().__init__(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                caption="Python FPS Sandbox",
                resizable=True,
                config=config,
                visible=visible,
                vsync=True,
            )
        except pyglet.window.NoSuchConfigException:
            super().__init__(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                caption="Python FPS Sandbox",
                resizable=True,
                config=Config(double_buffer=True, depth_size=24),
                visible=visible,
                vsync=True,
            )

        glClearColor(0.56, 0.76, 0.92, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

        self.yaw = 0.0
        self.pitch = -0.08
        self.camera_position = Vec3(0.0, 1.8, 8.0)
        self.light_dir = (0.5, -1.0, 0.35)
        self.mouse_captured = False

        self.shader = self._build_shader()
        self.cube_mesh = build_cube_mesh(self.shader)
        self.plane_mesh = build_plane_mesh(self.shader)
        self.scene = self._build_scene()

        self.instructions = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=12,
            x=16,
            y=self.height - 16,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=max(self.width - 32, 280),
            color=(16, 24, 32, 255),
        )
        self.status = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=12,
            x=16,
            y=16,
            anchor_x="left",
            anchor_y="bottom",
            color=(16, 24, 32, 255),
        )
        self.crosshair = pyglet.text.Label(
            "+",
            font_name="Consolas",
            font_size=18,
            x=self.width // 2,
            y=self.height // 2,
            anchor_x="center",
            anchor_y="center",
            color=(12, 20, 28, 220),
        )
        self._refresh_labels()

        if not pyglet.options["headless"]:
            self.set_exclusive_mouse(True)
            self.mouse_captured = True

        auto_close = float(os.getenv("FPS_DEMO_AUTOCLOSE_SECONDS", "0") or "0")
        if auto_close > 0:
            pyglet.clock.schedule_once(lambda _dt: self.close(), auto_close)

        pyglet.clock.schedule_interval(self.update, 1 / 120)

    def _build_shader(self) -> ShaderProgram:
        vertex = Shader(VERTEX_SHADER, "vertex")
        fragment = Shader(FRAGMENT_SHADER, "fragment")
        program = ShaderProgram(vertex, fragment)
        self._update_projection()
        program["light_dir"] = self.light_dir
        return program

    def _build_scene(self) -> list[SceneObject]:
        scene = [
            SceneObject(
                mesh=self.plane_mesh,
                position=Vec3(0.0, -0.5, 0.0),
                scale=Vec3(120.0, 1.0, 120.0),
                tint=(0.64, 0.71, 0.66, 1.0),
                grid_strength=0.5,
            ),
            SceneObject(
                mesh=self.cube_mesh,
                position=Vec3(0.0, 1.0, 0.0),
                scale=Vec3(2.0, 2.0, 2.0),
                tint=(0.91, 0.52, 0.36, 1.0),
            ),
            SceneObject(
                mesh=self.cube_mesh,
                position=Vec3(0.0, 4.0, -12.0),
                scale=Vec3(6.0, 0.5, 6.0),
                tint=(0.88, 0.80, 0.46, 1.0),
            ),
        ]

        columns = [
            (-12.0, 3.0, -12.0),
            (12.0, 3.0, -12.0),
            (-12.0, 3.0, 12.0),
            (12.0, 3.0, 12.0),
            (-18.0, 4.5, 0.0),
            (18.0, 4.5, 0.0),
        ]
        for x, y, z in columns:
            scene.append(
                SceneObject(
                    mesh=self.cube_mesh,
                    position=Vec3(x, y, z),
                    scale=Vec3(2.0, y * 2.0, 2.0),
                    tint=(0.44, 0.62, 0.82, 1.0),
                )
            )

        crate_positions = [
            (-8.0, 1.0, -4.0),
            (-4.0, 0.5, -10.0),
            (5.0, 1.0, -8.0),
            (9.0, 1.5, 2.0),
            (-6.0, 1.5, 6.0),
            (14.0, 0.75, -2.0),
            (3.0, 2.5, -16.0),
            (-15.0, 2.0, 8.0),
        ]
        crate_scales = [
            Vec3(1.5, 1.5, 1.5),
            Vec3(1.0, 1.0, 1.0),
            Vec3(2.0, 2.0, 2.0),
            Vec3(3.0, 3.0, 3.0),
            Vec3(2.0, 3.0, 2.0),
            Vec3(1.5, 1.5, 1.5),
            Vec3(2.5, 5.0, 2.5),
            Vec3(4.0, 4.0, 4.0),
        ]
        crate_colors = [
            (0.78, 0.41, 0.36, 1.0),
            (0.64, 0.51, 0.84, 1.0),
            (0.38, 0.70, 0.58, 1.0),
            (0.93, 0.72, 0.34, 1.0),
            (0.52, 0.62, 0.88, 1.0),
            (0.86, 0.57, 0.49, 1.0),
            (0.57, 0.74, 0.74, 1.0),
            (0.73, 0.46, 0.66, 1.0),
        ]
        for position, scale, tint in zip(crate_positions, crate_scales, crate_colors, strict=True):
            scene.append(
                SceneObject(
                    mesh=self.cube_mesh,
                    position=Vec3(*position),
                    scale=scale,
                    tint=tint,
                )
            )

        return scene

    def _update_projection(self) -> None:
        aspect = self.width / max(self.height, 1)
        self.projection = Mat4.perspective_projection(aspect, 0.1, 250.0, 75.0)

    def build_view_matrix(self) -> Mat4:
        direction = self._forward_vector()
        target = self.camera_position + direction
        return Mat4.look_at(self.camera_position, target, Vec3(0.0, 1.0, 0.0))

    def _forward_vector(self) -> Vec3:
        cos_pitch = math.cos(self.pitch)
        return Vec3(
            math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cos_pitch,
        ).normalize()

    def _ground_forward_vector(self) -> Vec3:
        return Vec3(math.sin(self.yaw), 0.0, -math.cos(self.yaw)).normalize()

    def _right_vector(self) -> Vec3:
        return Vec3(math.cos(self.yaw), 0.0, math.sin(self.yaw)).normalize()

    def _refresh_labels(self) -> None:
        self.instructions.text = (
            "WASD move   SPACE / CTRL rise-fall   SHIFT sprint\n"
            "Mouse look   TAB toggle cursor capture   ESC release or close\n"
            "Left click recaptures the mouse   R resets the start position"
        )
        self.instructions.width = max(self.width - 32, 280)
        self.instructions.x = 16
        self.instructions.y = self.height - 16

        self.crosshair.x = self.width // 2
        self.crosshair.y = self.height // 2

    def set_capture(self, enabled: bool) -> None:
        self.mouse_captured = enabled
        self.set_exclusive_mouse(enabled)

    def reset_camera(self) -> None:
        self.camera_position = Vec3(0.0, 1.8, 8.0)
        self.yaw = 0.0
        self.pitch = -0.08

    def on_resize(self, width: int, height: int):
        glViewport(0, 0, width, height)
        self._update_projection()
        self._refresh_labels()
        return super().on_resize(width, height)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == mouse.LEFT and not self.mouse_captured and not pyglet.options["headless"]:
            self.set_capture(True)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if not self.mouse_captured:
            return
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch = max(-1.54, min(1.54, self.pitch + dy * MOUSE_SENSITIVITY))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.TAB and not pyglet.options["headless"]:
            self.set_capture(not self.mouse_captured)
        elif symbol == key.ESCAPE:
            if self.mouse_captured and not pyglet.options["headless"]:
                self.set_capture(False)
            else:
                self.close()
        elif symbol == key.R:
            self.reset_camera()

    def update(self, dt: float) -> None:
        move = Vec3(0.0, 0.0, 0.0)
        vertical = 0.0
        ground_forward = self._ground_forward_vector()
        right = self._right_vector()

        if self.keys[key.W]:
            move += ground_forward
        if self.keys[key.S]:
            move -= ground_forward
        if self.keys[key.D]:
            move += right
        if self.keys[key.A]:
            move -= right
        if self.keys[key.SPACE]:
            vertical += 1.0
        if self.keys[key.LCTRL] or self.keys[key.RCTRL]:
            vertical -= 1.0

        if move.length() > 0:
            move = move.normalize()

        horizontal_speed = SPRINT_SPEED if self.keys[key.LSHIFT] or self.keys[key.RSHIFT] else MOVE_SPEED
        self.camera_position += move * horizontal_speed * dt
        self.camera_position += Vec3(0.0, vertical * VERTICAL_SPEED * dt, 0.0)

        self.status.text = (
            f"pos=({self.camera_position.x:6.2f}, {self.camera_position.y:5.2f}, {self.camera_position.z:6.2f})   "
            f"yaw={math.degrees(self.yaw):6.1f}   pitch={math.degrees(self.pitch):5.1f}"
        )

    def on_draw(self) -> None:
        self.clear()
        self.shader.use()
        self.shader["projection"] = self.projection
        self.shader["view"] = self.build_view_matrix()

        for obj in self.scene:
            self.shader["model"] = obj.model_matrix()
            self.shader["tint"] = obj.tint
            self.shader["grid_strength"] = obj.grid_strength
            obj.mesh.draw(GL_TRIANGLES)

        glDisable(GL_DEPTH_TEST)
        self.instructions.draw()
        self.status.draw()
        if self.mouse_captured:
            self.crosshair.draw()
        glEnable(GL_DEPTH_TEST)


def main() -> None:
    FpsSandboxWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
