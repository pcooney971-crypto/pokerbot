# Build Instructions (PyInstaller)

## 1) Install dependencies

```bash
python -m pip install -r requirements.txt pyinstaller
```

## 2) Build a standalone executable

### Windows (.exe)

```bash
pyinstaller --noconfirm --onefile --windowed --name PokerHUD --collect-all ultralytics --hidden-import PySide6.QtSvg --hidden-import PySide6.QtOpenGL poker_hud.py
```

### macOS (.app)

```bash
pyinstaller --noconfirm --windowed --name PokerHUD --collect-all ultralytics --hidden-import PySide6.QtSvg --hidden-import PySide6.QtOpenGL poker_hud.py
```

> Output artifacts are created in `dist/`.

## 3) Run the app

```bash
./dist/PokerHUD --model /absolute/path/to/your_model.pt
```

(Use `PokerHUD.exe` on Windows.)
