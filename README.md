# 3D Game Develop

Python first-person 3D sandbox prototype.

The current build provides:

- A start screen with menu buttons
- A free-look first-person camera
- Block collision and grounded movement
- Gravity, jumping, and glide descent
- A simple 3D test space with a ground grid and landmark objects
- A minimal structure that can grow into a larger game

## Requirements

- Python 3.14 or compatible
- `pyglet` 2.1.13

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The game opens on the title screen first.

## Controls

- Title screen buttons: `게임 시작`, `게임 방법`, `게임 종료`
- `W`, `A`, `S`, `D`: move
- `Mouse`: look around
- `Space`: jump
- Release `Space`, then hold it again in midair to glide and fall slowly
- `Shift`: sprint
- `Tab`: toggle mouse capture during play
- `Esc`: open or close the menu
- `R`: reset camera position

## Next Steps

- Replace placeholder geometry with modular level pieces
- Add interaction, UI, and game rules
- Add simple enemies or pickups to validate gameplay flow
