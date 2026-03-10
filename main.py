from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

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
WINDOW_TITLE = "점프맵"
MENU_FONT = "Malgun Gothic"
HUD_FONT = "Consolas"
MOUSE_SENSITIVITY = 0.0025
MOVE_SPEED = 8.0
SPRINT_SPEED = 14.0
JUMP_SPEED = 11.0
GRAVITY = 30.0
GLIDE_GRAVITY = 7.5
MAX_FALL_SPEED = 25.0
GLIDE_FALL_SPEED = 4.0
PLAYER_RADIUS = 0.35
PLAYER_HEIGHT = 1.8
PLAYER_EYE_HEIGHT = 1.62

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
class Aabb:
    minimum: Vec3
    maximum: Vec3

    def intersects(self, other: "Aabb") -> bool:
        return (
            self.minimum.x < other.maximum.x
            and self.maximum.x > other.minimum.x
            and self.minimum.y < other.maximum.y
            and self.maximum.y > other.minimum.y
            and self.minimum.z < other.maximum.z
            and self.maximum.z > other.minimum.z
        )


@dataclass(frozen=True)
class SceneObject:
    mesh: object
    position: Vec3
    scale: Vec3
    tint: tuple[float, float, float, float]
    rotation_y: float = 0.0
    grid_strength: float = 0.0
    collidable: bool = True

    def model_matrix(self) -> Mat4:
        matrix = Mat4().translate(self.position)
        if self.rotation_y:
            matrix = matrix.rotate(self.rotation_y, Vec3(0.0, 1.0, 0.0))
        return matrix.scale(self.scale)

    def bounds(self) -> Aabb:
        half_scale = Vec3(self.scale.x * 0.5, self.scale.y * 0.5, self.scale.z * 0.5)
        return Aabb(self.position - half_scale, self.position + half_scale)


@dataclass
class MenuButton:
    text: str
    action: Callable[[], None]
    rect: pyglet.shapes.Rectangle
    label: pyglet.text.Label
    hovered: bool = False

    def contains(self, x: float, y: float) -> bool:
        return self.rect.x <= x <= self.rect.x + self.rect.width and self.rect.y <= y <= self.rect.y + self.rect.height

    def set_bounds(self, x: float, y: float, width: float, height: float) -> None:
        self.rect.x = x
        self.rect.y = y
        self.rect.width = width
        self.rect.height = height
        self.label.x = x + width / 2
        self.label.y = y + height / 2

    def set_hovered(self, hovered: bool) -> None:
        self.hovered = hovered
        self.rect.color = (38, 140, 148) if hovered else (32, 60, 76)
        self.rect.opacity = 235 if hovered else 215
        self.label.color = (248, 252, 255, 255) if hovered else (227, 236, 241, 255)

    def draw(self) -> None:
        self.rect.draw()
        self.label.draw()


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
                caption=WINDOW_TITLE,
                resizable=True,
                config=config,
                visible=visible,
                vsync=True,
            )
        except pyglet.window.NoSuchConfigException:
            super().__init__(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                caption=WINDOW_TITLE,
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
        self.player_position = Vec3(0.0, 0.0, 8.0)
        self.camera_position = Vec3(0.0, PLAYER_EYE_HEIGHT, 8.0)
        self.vertical_velocity = 0.0
        self.is_grounded = True
        self.jump_requested = False
        self.glide_ready = False
        self.is_gliding = False
        self.light_dir = (0.5, -1.0, 0.35)
        self.mouse_captured = False
        self.game_started = False
        self.menu_state: str | None = "main"
        self.menu_hover_button: MenuButton | None = None

        self.shader = self._build_shader()
        self.cube_mesh = build_cube_mesh(self.shader)
        self.plane_mesh = build_plane_mesh(self.shader)
        self.scene = self._build_scene()
        self.colliders = [obj for obj in self.scene if obj.collidable]

        self.instructions = pyglet.text.Label(
            "",
            font_name=HUD_FONT,
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
            font_name=HUD_FONT,
            font_size=12,
            x=16,
            y=16,
            anchor_x="left",
            anchor_y="bottom",
            color=(16, 24, 32, 255),
        )
        self.crosshair = pyglet.text.Label(
            "+",
            font_name=HUD_FONT,
            font_size=18,
            x=self.width // 2,
            y=self.height // 2,
            anchor_x="center",
            anchor_y="center",
            color=(12, 20, 28, 220),
        )
        self._build_menu_ui()
        self._refresh_labels()
        self.open_menu("main")

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

    def _build_menu_ui(self) -> None:
        self.menu_overlay = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(8, 14, 20))
        self.menu_overlay.opacity = 160
        self.menu_panel = pyglet.shapes.Rectangle(0, 0, 520, 420, color=(241, 233, 220))
        self.menu_panel.opacity = 244
        self.menu_accent = pyglet.shapes.Rectangle(0, 0, 520, 10, color=(32, 132, 140))
        self.menu_accent.opacity = 255
        self.menu_title = pyglet.text.Label(
            "점프맵",
            font_name=MENU_FONT,
            font_size=38,
            x=0,
            y=0,
            anchor_x="left",
            anchor_y="baseline",
            color=(25, 34, 49, 255),
        )
        self.menu_subtitle = pyglet.text.Label(
            "점프와 글라이딩으로 블럭 사이를 돌파하는 3D 프로토타입",
            font_name=MENU_FONT,
            font_size=13,
            x=0,
            y=0,
            width=420,
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            color=(89, 94, 104, 255),
        )
        self.help_title = pyglet.text.Label(
            "게임 방법",
            font_name=MENU_FONT,
            font_size=27,
            x=0,
            y=0,
            anchor_x="left",
            anchor_y="baseline",
            color=(25, 34, 49, 255),
        )
        self.help_body = pyglet.text.Label(
            "움직이기\n"
            "W A S D : 이동\n"
            "마우스 : 시점 회전\n"
            "Shift : 빠르게 이동\n\n"
            "액션\n"
            "Space : 점프\n"
            "공중에서 Space를 놓았다가\n"
            "다시 누르고 유지 : 글라이딩\n"
            "R : 시작 위치로 리셋\n"
            "ESC : 메뉴 열기 / 닫기",
            font_name=MENU_FONT,
            font_size=14,
            x=0,
            y=0,
            width=420,
            multiline=True,
            anchor_x="left",
            anchor_y="top",
            color=(57, 67, 78, 255),
        )
        self.main_menu_buttons = [
            self._create_menu_button("게임 시작", self.start_game),
            self._create_menu_button("게임 방법", self.show_help_menu),
            self._create_menu_button("게임 종료", self.close),
        ]
        self.help_menu_buttons = [
            self._create_menu_button("뒤로", self.show_main_menu),
            self._create_menu_button("바로 시작", self.start_game),
        ]
        self._layout_menu_ui()

    def _create_menu_button(self, text: str, action: Callable[[], None]) -> MenuButton:
        rect = pyglet.shapes.Rectangle(0, 0, 100, 50, color=(32, 60, 76))
        rect.opacity = 215
        label = pyglet.text.Label(
            text,
            font_name=MENU_FONT,
            font_size=16,
            x=0,
            y=0,
            anchor_x="center",
            anchor_y="center",
            color=(227, 236, 241, 255),
        )
        button = MenuButton(text=text, action=action, rect=rect, label=label)
        button.set_hovered(False)
        return button

    def _layout_menu_ui(self) -> None:
        self.menu_overlay.width = self.width
        self.menu_overlay.height = self.height

        panel_width = min(560, max(self.width - 96, 360))
        panel_height = 500 if self.menu_state == "help" else 420
        panel_x = (self.width - panel_width) / 2
        panel_y = (self.height - panel_height) / 2

        self.menu_panel.x = panel_x
        self.menu_panel.y = panel_y
        self.menu_panel.width = panel_width
        self.menu_panel.height = panel_height

        self.menu_accent.x = panel_x
        self.menu_accent.y = panel_y + panel_height - 14
        self.menu_accent.width = panel_width

        self.menu_title.x = panel_x + 42
        self.menu_title.y = panel_y + panel_height - 72
        self.menu_subtitle.x = panel_x + 42
        self.menu_subtitle.y = panel_y + panel_height - 100
        self.menu_subtitle.width = panel_width - 84

        self.help_title.x = panel_x + 42
        self.help_title.y = panel_y + panel_height - 72
        self.help_body.x = panel_x + 42
        self.help_body.y = panel_y + panel_height - 108
        self.help_body.width = panel_width - 84

        main_button_width = panel_width - 84
        main_button_x = panel_x + 42
        main_button_height = 56
        main_button_gap = 16
        main_button_y = panel_y + 56
        for index, button in enumerate(reversed(self.main_menu_buttons)):
            button.set_bounds(
                main_button_x,
                main_button_y + index * (main_button_height + main_button_gap),
                main_button_width,
                main_button_height,
            )

        help_button_width = (panel_width - 98) / 2
        help_button_y = panel_y + 38
        self.help_menu_buttons[0].set_bounds(panel_x + 42, help_button_y, help_button_width, 52)
        self.help_menu_buttons[1].set_bounds(panel_x + 56 + help_button_width, help_button_y, help_button_width, 52)

        self._update_menu_button_labels()
        self._update_menu_hover(-1, -1)

    def _update_menu_button_labels(self) -> None:
        start_text = "계속하기" if self.game_started else "게임 시작"
        self.main_menu_buttons[0].label.text = start_text
        self.main_menu_buttons[0].text = start_text
        help_start_text = "바로 시작" if not self.game_started else "게임으로 돌아가기"
        self.help_menu_buttons[1].label.text = help_start_text
        self.help_menu_buttons[1].text = help_start_text

    def _current_menu_buttons(self) -> list[MenuButton]:
        if self.menu_state == "help":
            return self.help_menu_buttons
        if self.menu_state == "main":
            return self.main_menu_buttons
        return []

    def _update_menu_hover(self, x: float, y: float) -> None:
        hovered_button = None
        for button in self.main_menu_buttons + self.help_menu_buttons:
            button.set_hovered(False)

        for button in self._current_menu_buttons():
            if button.contains(x, y):
                button.set_hovered(True)
                hovered_button = button
                break

        self.menu_hover_button = hovered_button

    def open_menu(self, state: str = "main") -> None:
        self.menu_state = state
        if self.mouse_captured and not pyglet.options["headless"]:
            self.set_capture(False)
        self._layout_menu_ui()

    def show_main_menu(self) -> None:
        self.open_menu("main")

    def show_help_menu(self) -> None:
        self.open_menu("help")

    def start_game(self) -> None:
        if not self.game_started:
            self.reset_camera()
            self.game_started = True
        self.menu_state = None
        self._update_menu_button_labels()
        if not pyglet.options["headless"]:
            self.set_capture(True)

    def _update_projection(self) -> None:
        aspect = self.width / max(self.height, 1)
        self.world_projection = Mat4.perspective_projection(aspect, 0.1, 250.0, 75.0)

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

    def _player_bounds(self, position: Vec3 | None = None) -> Aabb:
        player_position = self.player_position if position is None else position
        return Aabb(
            Vec3(player_position.x - PLAYER_RADIUS, player_position.y, player_position.z - PLAYER_RADIUS),
            Vec3(player_position.x + PLAYER_RADIUS, player_position.y + PLAYER_HEIGHT, player_position.z + PLAYER_RADIUS),
        )

    def _update_camera_position(self) -> None:
        self.camera_position = Vec3(
            self.player_position.x,
            self.player_position.y + PLAYER_EYE_HEIGHT,
            self.player_position.z,
        )

    def _move_player_axis(self, amount: float, axis: str) -> bool:
        if amount == 0.0:
            return False

        if axis == "x":
            candidate = Vec3(self.player_position.x + amount, self.player_position.y, self.player_position.z)
        elif axis == "y":
            candidate = Vec3(self.player_position.x, self.player_position.y + amount, self.player_position.z)
        else:
            candidate = Vec3(self.player_position.x, self.player_position.y, self.player_position.z + amount)

        landed = False
        for collider in self.colliders:
            collider_bounds = collider.bounds()
            player_bounds = self._player_bounds(candidate)
            if not player_bounds.intersects(collider_bounds):
                continue

            if axis == "x":
                if amount > 0.0:
                    candidate = Vec3(collider_bounds.minimum.x - PLAYER_RADIUS, candidate.y, candidate.z)
                else:
                    candidate = Vec3(collider_bounds.maximum.x + PLAYER_RADIUS, candidate.y, candidate.z)
            elif axis == "z":
                if amount > 0.0:
                    candidate = Vec3(candidate.x, candidate.y, collider_bounds.minimum.z - PLAYER_RADIUS)
                else:
                    candidate = Vec3(candidate.x, candidate.y, collider_bounds.maximum.z + PLAYER_RADIUS)
            else:
                if amount > 0.0:
                    candidate = Vec3(candidate.x, collider_bounds.minimum.y - PLAYER_HEIGHT, candidate.z)
                else:
                    candidate = Vec3(candidate.x, collider_bounds.maximum.y, candidate.z)
                    landed = True
                self.vertical_velocity = 0.0

        self.player_position = candidate
        return landed

    def _refresh_labels(self) -> None:
        self.instructions.text = (
            "WASD move   SPACE jump   SHIFT sprint\n"
            "Release SPACE, then hold it in midair to glide\n"
            "Mouse look   TAB toggle cursor capture   ESC menu\n"
            "Left click recaptures the mouse   R resets the start position"
        )
        self.instructions.width = max(self.width - 32, 280)
        self.instructions.x = 16
        self.instructions.y = self.height - 16

        self.crosshair.x = self.width // 2
        self.crosshair.y = self.height // 2
        self._layout_menu_ui()

    def set_capture(self, enabled: bool) -> None:
        self.mouse_captured = enabled
        self.set_exclusive_mouse(enabled)

    def reset_camera(self) -> None:
        self.player_position = Vec3(0.0, 0.0, 8.0)
        self.yaw = 0.0
        self.pitch = -0.08
        self.vertical_velocity = 0.0
        self.is_grounded = True
        self.jump_requested = False
        self.glide_ready = False
        self.is_gliding = False
        self._update_camera_position()

    def on_resize(self, width: int, height: int):
        glViewport(0, 0, width, height)
        self._update_projection()
        self._refresh_labels()
        return super().on_resize(width, height)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.menu_state is not None and button == mouse.LEFT:
            for menu_button in self._current_menu_buttons():
                if menu_button.contains(x, y):
                    menu_button.action()
                    return
            return
        if button == mouse.LEFT and not self.mouse_captured and not pyglet.options["headless"]:
            self.set_capture(True)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if self.menu_state is not None:
            self._update_menu_hover(x, y)
            return
        if not self.mouse_captured:
            return
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch = max(-1.54, min(1.54, self.pitch + dy * MOUSE_SENSITIVITY))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self.menu_state == "help":
            if symbol == key.ESCAPE:
                self.show_main_menu()
            elif symbol in (key.ENTER, key.SPACE):
                self.start_game()
            return

        if self.menu_state == "main":
            if symbol == key.ESCAPE:
                if self.game_started:
                    self.start_game()
            elif symbol in (key.ENTER, key.SPACE):
                self.start_game()
            return

        if symbol == key.TAB and not pyglet.options["headless"]:
            self.set_capture(not self.mouse_captured)
        elif symbol == key.ESCAPE:
            self.show_main_menu()
        elif symbol == key.SPACE:
            self.jump_requested = True
        elif symbol == key.R:
            self.reset_camera()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        if symbol == key.SPACE:
            if not self.is_grounded:
                self.glide_ready = True
            self.is_gliding = False

    def update(self, dt: float) -> None:
        if self.menu_state is not None:
            return

        dt = min(dt, 0.05)
        move = Vec3(0.0, 0.0, 0.0)
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

        if move.length() > 0:
            move = move.normalize()

        if self.jump_requested and self.is_grounded:
            self.vertical_velocity = JUMP_SPEED
            self.is_grounded = False
            self.glide_ready = False
            self.is_gliding = False
        self.jump_requested = False

        glide_pressed = self.keys[key.SPACE] and not self.is_grounded
        self.is_gliding = self.glide_ready and glide_pressed and self.vertical_velocity < 0.0
        gravity = GLIDE_GRAVITY if self.is_gliding else GRAVITY
        fall_speed_limit = GLIDE_FALL_SPEED if self.is_gliding else MAX_FALL_SPEED
        self.vertical_velocity = max(self.vertical_velocity - gravity * dt, -fall_speed_limit)
        horizontal_speed = SPRINT_SPEED if self.keys[key.LSHIFT] or self.keys[key.RSHIFT] else MOVE_SPEED
        horizontal_move = move * horizontal_speed * dt
        self._move_player_axis(horizontal_move.x, "x")
        self._move_player_axis(horizontal_move.z, "z")
        self.is_grounded = self._move_player_axis(self.vertical_velocity * dt, "y")
        if self.is_grounded:
            self.glide_ready = False
            self.is_gliding = False
        self._update_camera_position()

        self.status.text = (
            f"pos=({self.camera_position.x:6.2f}, {self.camera_position.y:5.2f}, {self.camera_position.z:6.2f})   "
            f"yaw={math.degrees(self.yaw):6.1f}   pitch={math.degrees(self.pitch):5.1f}   "
            f"grounded={'yes' if self.is_grounded else 'no '}   "
            f"glide={'on ' if self.is_gliding else 'off'}   vy={self.vertical_velocity:6.2f}"
        )

    def on_draw(self) -> None:
        self.clear()
        self.shader.use()
        self.shader["projection"] = self.world_projection
        self.shader["view"] = self.build_view_matrix()

        for obj in self.scene:
            self.shader["model"] = obj.model_matrix()
            self.shader["tint"] = obj.tint
            self.shader["grid_strength"] = obj.grid_strength
            obj.mesh.draw(GL_TRIANGLES)

        self.shader.stop()
        glDisable(GL_DEPTH_TEST)
        if self.menu_state is None:
            self.instructions.draw()
            self.status.draw()
            if self.mouse_captured:
                self.crosshair.draw()
        else:
            self.menu_overlay.draw()
            self.menu_panel.draw()
            self.menu_accent.draw()
            if self.menu_state == "help":
                self.help_title.draw()
                self.help_body.draw()
            else:
                self.menu_title.draw()
                self.menu_subtitle.draw()
            for button in self._current_menu_buttons():
                button.draw()
        glEnable(GL_DEPTH_TEST)


def main() -> None:
    FpsSandboxWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
