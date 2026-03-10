# 3D Game Develop

Python first-person 3D sandbox prototype.

The current build provides:

- A free-look first-person camera
- Block collision and grounded movement
- Gravity and jumping
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

## Controls

- `W`, `A`, `S`, `D`: move
- `Mouse`: look around
- `Space`: jump
- `Shift`: sprint
- `Tab`: toggle mouse capture
- `Esc`: release mouse or close the game
- `R`: reset camera position

## Next Steps

- Replace placeholder geometry with modular level pieces
- Add interaction, UI, and game rules
- Add simple enemies or pickups to validate gameplay flow
