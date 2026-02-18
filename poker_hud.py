#!/usr/bin/env python3
"""Poker HUD desktop app (macOS/Windows/Linux).

- Zero-calibration full-screen detection with YOLOv8 (ultralytics).
- Transparent, frameless, always-on-top click-through overlay.
- Separate clickable control panel.
- Real-time win equity + action recommendation.
- Works as a double-clickable app bundle (model can be selected in UI).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from treys import Card, Evaluator
from ultralytics import YOLO

CARD_RE = re.compile(r"(10|[2-9TJQKA])([shdcSHDC♠♥♦♣])")
RANK_MAP = {"10": "T", "T": "T", "J": "J", "Q": "Q", "K": "K", "A": "A"}
SUIT_MAP = {"s": "s", "h": "h", "d": "d", "c": "c", "♠": "s", "♥": "h", "♦": "d", "♣": "c"}


@dataclass
class DetectionItem:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


@dataclass
class HudState:
    player_cards: List[str] = field(default_factory=list)
    community_cards: List[str] = field(default_factory=list)
    pot_size: Optional[str] = None
    win_probability: Optional[float] = None
    best_move: str = "Waiting for cards..."
    text_anchor: Optional[Tuple[int, int]] = None
    debug_summary: str = ""


class EquityCalculator:
    def __init__(self) -> None:
        self.evaluator = Evaluator()

    def estimate(
        self,
        hero_cards: Sequence[str],
        board_cards: Sequence[str],
        opponents: int = 1,
        samples: int = 3000,
    ) -> float:
        if len(hero_cards) != 2:
            return 0.0

        try:
            hero = [Card.new(c) for c in hero_cards]
            board = [Card.new(c) for c in board_cards]
        except Exception:
            return 0.0

        used = set(hero + board)
        deck = [
            Card.new(r + s)
            for r in "23456789TJQKA"
            for s in "shdc"
            if Card.new(r + s) not in used
        ]
        needed = (2 * opponents) + (5 - len(board))
        if len(deck) < needed:
            return 0.0

        wins = 0.0
        for _ in range(samples):
            sample = random.sample(deck, needed)
            idx = 0
            villain_hands: List[List[int]] = []
            for _op in range(opponents):
                villain_hands.append([sample[idx], sample[idx + 1]])
                idx += 2

            full_board = board + sample[idx:]
            hero_score = self.evaluator.evaluate(full_board, hero)
            opp_scores = [self.evaluator.evaluate(full_board, hand) for hand in villain_hands]
            best_opp = min(opp_scores)

            if hero_score < best_opp:
                wins += 1.0
            elif hero_score == best_opp:
                ties = sum(1 for score in opp_scores if score == hero_score)
                wins += 1.0 / (ties + 1)

        return wins / float(samples)

    @staticmethod
    def recommend_move(win_prob: float) -> str:
        if win_prob < 0.35:
            return "Fold"
        if win_prob < 0.60:
            return "Call"
        return "Raise"


class DetectorWorker(QtCore.QThread):
    state_updated = QtCore.Signal(object)
    status = QtCore.Signal(str)

    def __init__(self, model_path: str, confidence: float = 0.45, interval_ms: int = 180) -> None:
        super().__init__()
        self.model_path = model_path
        self.confidence = confidence
        self.interval_ms = interval_ms
        self._running = threading.Event()
        self._running.set()
        self._equity = EquityCalculator()

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        try:
            model = YOLO(self.model_path)
            names: Dict[int, str] = model.names  # type: ignore[assignment]
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            return

        self.status.emit("Detector running")

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            while self._running.is_set():
                tick = time.time()
                frame = np.array(sct.grab(monitor))[:, :, :3]
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    result = model.predict(source=bgr, conf=self.confidence, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    time.sleep(0.25)
                    continue

                detections = self._extract_detections(result, names)
                state = self._build_state(detections, monitor["height"])
                self.state_updated.emit(state)

                elapsed_ms = (time.time() - tick) * 1000
                self.msleep(max(5, self.interval_ms - int(elapsed_ms)))

        self.status.emit("Detector stopped")

    def _extract_detections(self, result, names: Dict[int, str]) -> List[DetectionItem]:
        out: List[DetectionItem] = []
        if result.boxes is None:
            return out

        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_idx in zip(xyxy, confs, clss):
            out.append(
                DetectionItem(
                    label=str(names.get(cls_idx, f"class_{cls_idx}")),
                    confidence=float(conf),
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                )
            )
        return out

    def _build_state(self, items: Sequence[DetectionItem], screen_h: int) -> HudState:
        hero: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        board: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        pot_size: Optional[str] = None

        for item in items:
            label = item.label
            lowered = label.lower()

            if "pot" in lowered:
                pot_size = label
                continue

            card = self._extract_card_code(label)
            if not card:
                continue

            if any(k in lowered for k in ("hero", "player", "hole")):
                hero.append((card, item.box, item.confidence))
            elif any(k in lowered for k in ("community", "board", "flop", "turn", "river")):
                board.append((card, item.box, item.confidence))
            else:
                y_mid = (item.box[1] + item.box[3]) // 2
                if y_mid < (screen_h // 2):
                    board.append((card, item.box, item.confidence))
                else:
                    hero.append((card, item.box, item.confidence))

        hero = self._dedupe_cards(hero)
        board = self._dedupe_cards(board)

        state = HudState(
            player_cards=[c[0] for c in hero][:2],
            community_cards=[c[0] for c in board][:5],
            pot_size=pot_size,
            debug_summary=f"hero={len(hero)}, board={len(board)}",
        )

        if hero:
            x1, y1, _, _ = hero[0][1]
            state.text_anchor = (x1, max(24, y1 - 48))

        if len(state.player_cards) == 2:
            win = self._equity.estimate(state.player_cards, state.community_cards)
            state.win_probability = win
            state.best_move = self._equity.recommend_move(win)

        return state

    @staticmethod
    def _dedupe_cards(cards: Sequence[Tuple[str, Tuple[int, int, int, int], float]]) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        by_card: Dict[str, Tuple[str, Tuple[int, int, int, int], float]] = {}
        for item in cards:
            prev = by_card.get(item[0])
            if prev is None or item[2] > prev[2]:
                by_card[item[0]] = item
        return sorted(by_card.values(), key=lambda x: (x[1][1], x[1][0]))

    @staticmethod
    def _extract_card_code(label: str) -> Optional[str]:
        compact = label.replace("_", "").replace("-", "")
        match = CARD_RE.search(compact)
        if not match:
            return None

        rank_raw, suit_raw = match.groups()
        rank = RANK_MAP.get(rank_raw.upper(), rank_raw.upper())
        suit = SUIT_MAP.get(suit_raw, SUIT_MAP.get(suit_raw.lower(), ""))
        if not suit:
            return None

        return f"{rank}{suit}"


class HudOverlay(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.state = HudState()

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )

        geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(geometry)

    def update_state(self, state: HudState) -> None:
        self.state = state
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        anchor = self.state.text_anchor or (40, self.height() - 170)
        win_text = (
            f"Win %: {self.state.win_probability * 100:.1f}%"
            if self.state.win_probability is not None
            else "Win %: --"
        )
        move_text = f"Best Move: {self.state.best_move}"

        box = QtCore.QRect(anchor[0], anchor[1], 390, 126)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 145))
        painter.drawRoundedRect(box, 14, 14)

        painter.setPen(QtGui.QColor("#00FFC6"))
        painter.setFont(QtGui.QFont("Menlo", 17, QtGui.QFont.Bold))
        painter.drawText(box.adjusted(14, 12, -14, -65), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, win_text)

        move_color = "#F44336" if self.state.best_move == "Fold" else "#FFC107" if self.state.best_move == "Call" else "#76FF03"
        painter.setPen(QtGui.QColor(move_color))
        painter.setFont(QtGui.QFont("Menlo", 23, QtGui.QFont.Black))
        painter.drawText(box.adjusted(14, 48, -14, -15), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, move_text)

        painter.setPen(QtGui.QColor("#E0E0E0"))
        painter.setFont(QtGui.QFont("Menlo", 10))
        details = f"Hero: {' '.join(self.state.player_cards) or '--'} | Board: {' '.join(self.state.community_cards) or '--'}"
        if self.state.pot_size:
            details += f" | Pot: {self.state.pot_size}"
        painter.drawText(18, self.height() - 16, details)


class ControlPanel(QtWidgets.QWidget):
    start_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    model_selected = QtCore.Signal(str)

    def __init__(self, model_path: str = "") -> None:
        super().__init__()
        self._model_path = model_path

        self.setWindowTitle("Poker HUD Controls")
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setFixedSize(430, 220)

        layout = QtWidgets.QVBoxLayout(self)

        self.model_label = QtWidgets.QLabel("Model: (not selected)")
        self.model_label.setWordWrap(True)
        self.model_button = QtWidgets.QPushButton("Choose .pt model")

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start HUD")
        self.stop_btn = QtWidgets.QPushButton("Stop HUD")
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.model_label)
        layout.addWidget(self.model_button)
        layout.addLayout(row)
        layout.addWidget(self.status_label)

        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(bool(model_path))
        if model_path:
            self._set_model_text(model_path)

        self.model_button.clicked.connect(self._choose_model)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _choose_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO model", str(Path.home()), "YOLO model (*.pt)")
        if path:
            self._model_path = path
            self._set_model_text(path)
            self.start_btn.setEnabled(True)
            self.model_selected.emit(path)

    def _set_model_text(self, path: str) -> None:
        self.model_label.setText(f"Model: {path}")

    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_requested.emit()

    def _on_stop(self) -> None:
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(bool(self._model_path))
        self.stop_requested.emit()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class PokerHudApp(QtCore.QObject):
    def __init__(self, model_path: str, confidence: float, interval_ms: int) -> None:
        super().__init__()
        self.model_path = model_path
        self.confidence = confidence
        self.interval_ms = interval_ms

        self.overlay = HudOverlay()
        self.controls = ControlPanel(model_path=model_path)
        self.worker: Optional[DetectorWorker] = None

        self.controls.start_requested.connect(self.start)
        self.controls.stop_requested.connect(self.stop)
        self.controls.model_selected.connect(self._set_model_path)

    def _set_model_path(self, path: str) -> None:
        self.model_path = path

    def show(self) -> None:
        self.overlay.showFullScreen()
        self.controls.show()

    def start(self) -> None:
        if not self.model_path:
            self.controls.set_status("Please choose a .pt model first")
            return
        if self.worker and self.worker.isRunning():
            return

        self.worker = DetectorWorker(
            model_path=self.model_path,
            confidence=self.confidence,
            interval_ms=self.interval_ms,
        )
        self.worker.state_updated.connect(self.overlay.update_state)
        self.worker.status.connect(self.controls.set_status)
        self.worker.start()

    def stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1800)
        self.controls.stop_btn.setEnabled(False)
        self.controls.start_btn.setEnabled(bool(self.model_path))


def _default_model_path() -> str:
    """Find bundled model path if app was packaged with --add-data."""
    candidate_names = ("model.pt", "poker_model.pt")

    roots = [Path.cwd()]
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
        roots.append(Path(sys.executable).resolve().parent)

    for root in roots:
        for name in candidate_names:
            test_path = root / name
            if test_path.exists():
                return str(test_path)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poker HUD with YOLOv8 + transparent overlay")
    parser.add_argument("--model", default="", help="Path to YOLO model (.pt). Optional in app mode.")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    parser.add_argument("--interval-ms", type=int, default=180, help="Capture/inference interval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)

    model_path = args.model or _default_model_path()
    hud = PokerHudApp(model_path=model_path, confidence=args.conf, interval_ms=args.interval_ms)
    hud.show()

    if model_path:
        hud.start()
    else:
        hud.controls.set_status("Select a .pt model and click Start HUD")

    exit_code = app.exec()
    hud.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
