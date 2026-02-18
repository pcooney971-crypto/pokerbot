#!/usr/bin/env python3
"""
Standalone Poker HUD overlay.

Features:
- Full-screen capture with mss.
- Zero-calibration object detection via YOLOv8 (ultralytics).
- Transparent, always-on-top click-through HUD.
- Separate control panel for start/stop actions.
- Monte Carlo equity estimation from detected cards.

Usage:
    python poker_hud.py --model ./your_model.pt
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from treys import Card, Evaluator
from ultralytics import YOLO

CARD_RE = re.compile(r"(10|[2-9TJQKA])([shdcSHDC♠♥♦♣])")
RANK_MAP = {"T": "T", "10": "T", "J": "J", "Q": "Q", "K": "K", "A": "A"}
SUIT_MAP = {
    "s": "s",
    "h": "h",
    "d": "d",
    "c": "c",
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c",
}


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
    """Monte Carlo equity estimator for Texas Hold'em."""

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
        deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc" if Card.new(r + s) not in used]
        if len(deck) < (2 * opponents + (5 - len(board))):
            return 0.0

        wins = 0.0
        for _ in range(samples):
            sample = random.sample(deck, 2 * opponents + (5 - len(board)))
            idx = 0
            opp_hands = []
            for _opp in range(opponents):
                opp_hands.append([sample[idx], sample[idx + 1]])
                idx += 2
            full_board = board + sample[idx:]

            hero_score = self.evaluator.evaluate(full_board, hero)
            opp_scores = [self.evaluator.evaluate(full_board, hand) for hand in opp_hands]

            best_opp = min(opp_scores)
            if hero_score < best_opp:
                wins += 1.0
            elif hero_score == best_opp:
                ties = sum(1 for s in opp_scores if s == hero_score)
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
                start = time.time()
                frame = np.array(sct.grab(monitor))[:, :, :3]
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    result = model.predict(source=bgr, conf=self.confidence, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    time.sleep(0.25)
                    continue

                detections = self._extract_detections(result, names)
                hud_state = self._build_state(detections)
                self.state_updated.emit(hud_state)

                elapsed_ms = (time.time() - start) * 1000
                sleep_ms = max(5, self.interval_ms - int(elapsed_ms))
                self.msleep(sleep_ms)

        self.status.emit("Detector stopped")

    def _extract_detections(self, result, names: Dict[int, str]) -> List[DetectionItem]:
        items: List[DetectionItem] = []
        if result.boxes is None:
            return items

        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_idx in zip(xyxy, confs, clss):
            label = str(names.get(cls_idx, f"class_{cls_idx}"))
            items.append(
                DetectionItem(
                    label=label,
                    confidence=float(conf),
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                )
            )
        return items

    def _build_state(self, items: Sequence[DetectionItem]) -> HudState:
        player_cards: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        community_cards: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        pot_text = None

        for item in items:
            lowered = item.label.lower()

            if "pot" in lowered:
                pot_text = item.label
                continue

            card = self._extract_card_code(item.label)
            if not card:
                continue

            if any(tag in lowered for tag in ("player", "hole", "hero")):
                player_cards.append((card, item.box, item.confidence))
            elif any(tag in lowered for tag in ("community", "board", "flop", "turn", "river")):
                community_cards.append((card, item.box, item.confidence))
            else:
                # Fallback heuristic by Y position (top half = board, bottom half = player)
                y_center = (item.box[1] + item.box[3]) // 2
                if y_center < 540:
                    community_cards.append((card, item.box, item.confidence))
                else:
                    player_cards.append((card, item.box, item.confidence))

        player_cards = self._dedupe_cards(player_cards)
        community_cards = self._dedupe_cards(community_cards)

        state = HudState(
            player_cards=[c[0] for c in player_cards][:2],
            community_cards=[c[0] for c in community_cards][:5],
            pot_size=pot_text,
            debug_summary=f"Detected: hero={len(player_cards)}, board={len(community_cards)}",
        )

        if state.player_cards:
            x1, y1, _, _ = player_cards[0][1]
            state.text_anchor = (x1, max(20, y1 - 45))

        if len(state.player_cards) == 2:
            equity = self._equity.estimate(state.player_cards, state.community_cards)
            state.win_probability = equity
            state.best_move = self._equity.recommend_move(equity)

        return state

    @staticmethod
    def _dedupe_cards(cards: Sequence[Tuple[str, Tuple[int, int, int, int], float]]) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        by_label: Dict[str, Tuple[str, Tuple[int, int, int, int], float]] = {}
        for card in cards:
            prev = by_label.get(card[0])
            if prev is None or card[2] > prev[2]:
                by_label[card[0]] = card
        return sorted(by_label.values(), key=lambda c: (c[1][1], c[1][0]))

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

        self.setWindowTitle("Poker HUD Overlay")
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )

        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

    def update_state(self, state: HudState) -> None:
        self.state = state
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        anchor = self.state.text_anchor or (40, self.height() - 160)
        win = self.state.win_probability
        win_text = f"Win %: {win * 100:.1f}%" if win is not None else "Win %: --"
        move_text = f"Best Move: {self.state.best_move}"

        rect = QtCore.QRect(anchor[0], anchor[1], 360, 110)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 150))
        painter.drawRoundedRect(rect, 12, 12)

        painter.setPen(QtGui.QPen(QtGui.QColor("#00FFAA")))
        font1 = QtGui.QFont("Consolas", 16, QtGui.QFont.Bold)
        painter.setFont(font1)
        painter.drawText(rect.adjusted(14, 14, -14, -55), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, win_text)

        move_color = "#FF5252" if self.state.best_move == "Fold" else "#FFD54F" if self.state.best_move == "Call" else "#66FF66"
        painter.setPen(QtGui.QPen(QtGui.QColor(move_color)))
        font2 = QtGui.QFont("Consolas", 22, QtGui.QFont.Black)
        painter.setFont(font2)
        painter.drawText(rect.adjusted(14, 45, -14, -8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, move_text)

        painter.setPen(QtGui.QPen(QtGui.QColor("#DDDDDD")))
        font3 = QtGui.QFont("Consolas", 10)
        painter.setFont(font3)
        details = f"Hero: {' '.join(self.state.player_cards) or '--'} | Board: {' '.join(self.state.community_cards) or '--'}"
        painter.drawText(20, self.height() - 20, details)


class ControlPanel(QtWidgets.QWidget):
    start_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.setWindowTitle("Poker HUD Controls")
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setFixedSize(380, 150)

        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.model_label = QtWidgets.QLabel(f"Model: {model_path}")
        self.model_label.setWordWrap(True)

        button_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)

        layout.addWidget(self.model_label)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_requested.emit()

    def _on_stop(self) -> None:
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_requested.emit()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class PokerHudApp(QtCore.QObject):
    def __init__(self, model_path: str, confidence: float, interval_ms: int) -> None:
        super().__init__()
        self.overlay = HudOverlay()
        self.controls = ControlPanel(model_path)
        self.worker = DetectorWorker(model_path=model_path, confidence=confidence, interval_ms=interval_ms)

        self.worker.state_updated.connect(self.overlay.update_state)
        self.worker.status.connect(self.controls.set_status)

        self.controls.start_requested.connect(self.start)
        self.controls.stop_requested.connect(self.stop)

    def show(self) -> None:
        self.overlay.showFullScreen()
        self.controls.show()

    def start(self) -> None:
        if not self.worker.isRunning():
            self.worker = DetectorWorker(
                model_path=self.worker.model_path,
                confidence=self.worker.confidence,
                interval_ms=self.worker.interval_ms,
            )
            self.worker.state_updated.connect(self.overlay.update_state)
            self.worker.status.connect(self.controls.set_status)
            self.worker.start()

    def stop(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poker HUD with YOLOv8 + transparent Qt overlay")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 .pt model")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--interval-ms", type=int, default=180, help="Capture/inference interval in milliseconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)

    hud = PokerHudApp(model_path=args.model, confidence=args.conf, interval_ms=args.interval_ms)
    hud.show()
    hud.start()

    exit_code = app.exec()
    hud.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
