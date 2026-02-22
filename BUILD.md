# Build a Double-Clickable macOS App (`PokerHUD.app`)

This project is packaged as a native `.app` bundle so you can launch it by double-click.

## 1) Install build dependencies

```bash
python3 -m pip install -r requirements.txt pyinstaller
```

## 2) (Optional) Add your trained model to the repo root

If you put your model at `./model.pt`, the app can auto-load it on startup.

If no bundled model is found, the app opens and lets you choose a `.pt` file from the control panel.

## 3) Build `PokerHUD.app`

```bash
pyinstaller \
  --noconfirm \
  --windowed \
  --name PokerHUD \
  --osx-bundle-identifier com.pokerhud.desktop \
  --collect-all ultralytics \
  --hidden-import PySide6.QtSvg \
  --hidden-import PySide6.QtOpenGL \
  --add-data "model.pt:." \
  poker_hud.py
```

Output app:

- `dist/PokerHUD.app`

## 4) Installable DMG (drag-and-drop)

Create a DMG that users can open and drag `PokerHUD.app` into `Applications`:

```bash
rm -rf dist/dmg-root
mkdir -p dist/dmg-root
cp -R dist/PokerHUD.app dist/dmg-root/
ln -s /Applications dist/dmg-root/Applications
hdiutil create -volname "PokerHUD" -srcfolder dist/dmg-root -ov -format UDZO dist/PokerHUD.dmg
```

Output installer:

- `dist/PokerHUD.dmg`

## 5) Launch

- Directly: double-click `dist/PokerHUD.app`
- Installed: open `PokerHUD.dmg`, drag app to `Applications`, then double-click from Launchpad/Finder.

## 6) First-run macOS permissions

The app needs:

- **Screen Recording** permission (System Settings → Privacy & Security → Screen Recording).

Without this permission, capture/detection will not work.

## Optional: ad-hoc code signing (reduces Gatekeeper warnings)

```bash
codesign --force --deep --sign - dist/PokerHUD.app
```
