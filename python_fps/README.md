# Python FPS Sandbox

파이썬으로 실행하는 1인칭 3D 기본 공간이다. 자유 이동이 가능한 프로토타입이며, 바닥 그리드와 여러 기준 오브젝트를 배치해 공간감을 확인할 수 있다.

## 설치

```powershell
python -m pip install -r requirements.txt
```

## 실행

```powershell
python main.py
```

## 조작

- `WASD`: 이동
- `마우스`: 시점 회전
- `Space` / `Ctrl`: 위아래 이동
- `Shift`: 빠르게 이동
- `Tab`: 마우스 캡처 토글
- `Esc`: 마우스 해제 또는 종료
- `R`: 시작 위치로 리셋

## 참고

- 충돌 판정 없이 자유롭게 비행하듯 움직이는 형태다.
- 간단한 자동 종료 테스트가 필요하면 `FPS_DEMO_AUTOCLOSE_SECONDS=1` 환경 변수를 주고 실행하면 된다.
