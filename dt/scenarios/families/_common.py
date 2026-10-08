"""Shared helpers for family generators (palettes, IR node factories, box matching)."""
from __future__ import annotations

import random
from typing import Iterable, Optional

from dt.ir import Box, Color, Document, Fill, Node, TextStyle

LIGHT_BGS = ("#ffffff", "#fdf7fe", "#fef7ff", "#f8f9fa", "#fbfcfe", "#f6f8fc")
DARK_BGS = ("#141218", "#1f1f1f", "#202124", "#121212", "#1b1b1f")
LIGHT_INK = ("#1d1b20", "#202124", "#1f1f1f", "#49454f", "#444746")
DARK_INK = ("#e6e0e9", "#e8eaed", "#e3e3e3", "#cac4d0")
ACCENTS = ("#6750a4", "#7d5260", "#b3261e", "#0b57d0", "#146c2e", "#8c4a00", "#625b71", "#d93025")
DARK_ACCENTS = ("#d0bcff", "#efb8c8", "#f2b8b5", "#a8c7fa", "#6dd58c", "#ffb870")

WORDS = ("project", "update", "review", "weekly", "draft", "report", "travel", "plan", "invoice", "team", "launch",
         "notes", "design", "budget", "meeting", "summary", "photos", "trip", "order", "shipping", "account",
         "security", "reminder", "welcome", "results", "agenda", "feedback", "release", "tickets", "schedule",
         "garden", "recipe", "concert", "lease", "estimate", "workshop", "volunteer", "newsletter", "survey")
NAMES = ("Alex Kim", "Priya Natarajan", "Sam Ortega", "Jordan Lee", "Mei Chen", "Tomás Ruiz", "Aisha Bello",
         "Google Workspace", "Chris Novak", "Dana Whitfield", "Lena Fischer", "Ravi Patel", "Noah Smith",
         "Kai Yamamoto", "Grace Okafor", "Elena Petrova", "Team Calendar", "Billing", "Marco Rossi", "Ines Duarte")


def rng_for(seed: int, salt: str) -> random.Random:
    return random.Random(f"{salt}:{seed}")


def pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def phrase(rng: random.Random, n_lo: int, n_hi: int, cap: bool = True) -> str:
    words = [pick(rng, WORDS) for _ in range(rng.randint(n_lo, n_hi))]
    s = " ".join(words)
    return s[:1].upper() + s[1:] if cap else s


def text_node(s: str, x: float, y: float, size: float, color: str, weight: int = 400, lh: Optional[float] = None) -> Node:
    lh = lh or round(size * 1.4)
    return Node(type="text", name=f"text:{s[:24]}", box=Box(x, y, max(8.0, len(s) * size * 0.55), lh), text=s,
                text_style=TextStyle(family="Roboto", size=size, weight=weight, line_height=lh, color=Color.from_hex(color)))


def rect(name: str, box: Box, fill: Color | str, radius: float = 0.0) -> Node:
    return Node(type="rect", name=name, box=box, fills=[Fill.solid(fill)], radius=(radius,) * 4)


def matched(pred_nodes: Iterable[Node], box: Box, iou_thr: float, types: Optional[set[str]] = None) -> Optional[Node]:
    """Best pred node (by IoU, optionally restricted to ``types``) overlapping ``box`` at >= ``iou_thr``."""
    best, best_iou = None, iou_thr
    for n in pred_nodes:
        if types is not None and n.type not in types:
            continue
        iou = n.box.iou(box)
        if iou >= best_iou:
            best, best_iou = n, iou
    return best


def is_raster(n: Node) -> bool:
    return n.type == "image" and bool(n.image_ref or n.meta.get("rasterised"))


def doc_nodes(doc: Document) -> list[Node]:
    from dt.scenarios.measures import visible_nodes
    return visible_nodes(doc)
