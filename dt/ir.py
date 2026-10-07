"""Intermediate Representation (IR) for UI screens.

This is THE contract between every stage of the pipeline:

    screenshot --perceive--> Document(IR) --mapping--> Document(IR with component refs/tokens)
               --render--> HTML/PNG   --compare--> loss   --refine--> Document(IR)
               --export--> Figma build-plan JSON

Design rules:
  * Pure dataclasses, no third-party deps, JSON round-trippable (to_dict / from_dict).
  * All geometry is ABSOLUTE, in screenshot pixel space at DPR 1 (floats allowed).
  * A Node's `box` always encloses its children (renderers may clip to it).
  * `meta` is a free-form dict for perception evidence; nothing downstream may *require* it.
  * Everything optional has a sensible default so stages can construct partial nodes.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Iterator, Literal, Optional

NodeType = Literal["frame", "rect", "text", "image", "icon", "ellipse", "line", "instance", "vector"]
LayoutMode = Literal["none", "row", "column"]
Align = Literal["start", "center", "end", "stretch"]
Justify = Literal["start", "center", "end", "space-between"]
Sizing = Literal["fixed", "hug", "fill"]


# --------------------------------------------------------------------------- geometry
@dataclass
class Box:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def __post_init__(self) -> None:
        self.x, self.y, self.w, self.h = float(self.x), float(self.y), float(self.w), float(self.h)

    # derived
    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersect(self, o: "Box") -> "Box":
        x1, y1 = max(self.x, o.x), max(self.y, o.y)
        x2, y2 = min(self.x2, o.x2), min(self.y2, o.y2)
        if x2 <= x1 or y2 <= y1:
            return Box(0, 0, 0, 0)
        return Box(x1, y1, x2 - x1, y2 - y1)

    def union(self, o: "Box") -> "Box":
        x1, y1 = min(self.x, o.x), min(self.y, o.y)
        x2, y2 = max(self.x2, o.x2), max(self.y2, o.y2)
        return Box(x1, y1, x2 - x1, y2 - y1)

    def iou(self, o: "Box") -> float:
        inter = self.intersect(o).area
        if inter <= 0:
            return 0.0
        return inter / (self.area + o.area - inter)

    def contains(self, o: "Box", tol: float = 0.5) -> bool:
        return (
            o.x >= self.x - tol and o.y >= self.y - tol and o.x2 <= self.x2 + tol and o.y2 <= self.y2 + tol
        )

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def expand(self, d: float) -> "Box":
        return Box(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)

    def translate(self, dx: float, dy: float) -> "Box":
        return Box(self.x + dx, self.y + dy, self.w, self.h)

    def rounded(self) -> "Box":
        return Box(round(self.x), round(self.y), round(self.w), round(self.h))

    def as_int(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) integer pixel slice bounds (x1/y1 exclusive)."""
        return int(math.floor(self.x)), int(math.floor(self.y)), int(math.ceil(self.x2)), int(math.ceil(self.y2))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @staticmethod
    def from_dict(d: dict) -> "Box":
        return Box(float(d.get("x", 0)), float(d.get("y", 0)), float(d.get("w", 0)), float(d.get("h", 0)))


# --------------------------------------------------------------------------- color
def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB (0-255) -> CIE L*a*b* (D65). Pure python; good enough for ΔE76/ΔE94 use."""
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(X / xn), f(Y / yn), f(Z / zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


@dataclass
class Color:
    r: int = 0
    g: int = 0
    b: int = 0
    a: float = 1.0  # 0..1

    def __post_init__(self) -> None:
        self.r, self.g, self.b, self.a = int(self.r), int(self.g), int(self.b), float(self.a)

    def hex(self) -> str:
        return "#%02x%02x%02x" % (self.r, self.g, self.b)

    def css(self) -> str:
        if self.a >= 0.999:
            return self.hex()
        return f"rgba({self.r},{self.g},{self.b},{round(self.a, 3)})"

    @staticmethod
    def from_hex(s: str, a: float = 1.0) -> "Color":
        s = s.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) == 8:
            a = int(s[6:8], 16) / 255.0
            s = s[:6]
        return Color(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)

    @staticmethod
    def from_rgb(r: float, g: float, b: float, a: float = 1.0) -> "Color":
        return Color(int(round(r)), int(round(g)), int(round(b)), float(a))

    def lab(self) -> tuple[float, float, float]:
        return rgb_to_lab(self.r, self.g, self.b)

    def delta_e(self, o: "Color") -> float:
        """CIE76 ΔE. <1 imperceptible, <2.3 JND, >10 clearly different."""
        l1, a1, b1 = self.lab()
        l2, a2, b2 = o.lab()
        return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

    def luminance(self) -> float:
        return 0.2126 * _srgb_to_linear(self.r) + 0.7152 * _srgb_to_linear(self.g) + 0.0722 * _srgb_to_linear(self.b)

    def to_dict(self) -> dict:
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    @staticmethod
    def from_dict(d) -> "Color":
        if isinstance(d, str):
            return Color.from_hex(d)
        return Color(int(d.get("r", 0)), int(d.get("g", 0)), int(d.get("b", 0)), float(d.get("a", 1.0)))


# --------------------------------------------------------------------------- paint
@dataclass
class GradientStop:
    pos: float  # 0..1
    color: Color


@dataclass
class Fill:
    kind: Literal["solid", "linear", "radial", "image"] = "solid"
    color: Optional[Color] = None
    stops: list[GradientStop] = field(default_factory=list)
    angle: float = 180.0  # CSS degrees for linear gradients
    image_ref: Optional[str] = None  # path/URL/base64 ref for image fills
    opacity: float = 1.0

    @staticmethod
    def solid(c: Color | str) -> "Fill":
        return Fill(kind="solid", color=Color.from_hex(c) if isinstance(c, str) else c)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "opacity": self.opacity}
        if self.color is not None:
            d["color"] = self.color.to_dict()
        if self.stops:
            d["stops"] = [{"pos": s.pos, "color": s.color.to_dict()} for s in self.stops]
            d["angle"] = self.angle
        if self.image_ref:
            d["image_ref"] = self.image_ref
        return d

    @staticmethod
    def from_dict(d: dict) -> "Fill":
        return Fill(
            kind=d.get("kind", "solid"),
            color=Color.from_dict(d["color"]) if d.get("color") is not None else None,
            stops=[GradientStop(float(s["pos"]), Color.from_dict(s["color"])) for s in d.get("stops", [])],
            angle=float(d.get("angle", 180.0)),
            image_ref=d.get("image_ref"),
            opacity=float(d.get("opacity", 1.0)),
        )


@dataclass
class Stroke:
    color: Color = field(default_factory=Color)
    width: float = 1.0
    align: Literal["inside", "center", "outside"] = "inside"
    # per-side widths; None means uniform `width`
    sides: Optional[tuple[float, float, float, float]] = None  # top, right, bottom, left

    def __post_init__(self) -> None:
        self.width = float(self.width)
        if self.sides is not None:
            self.sides = tuple(float(v) for v in self.sides)

    def to_dict(self) -> dict:
        d = {"color": self.color.to_dict(), "width": self.width, "align": self.align}
        if self.sides is not None:
            d["sides"] = list(self.sides)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Stroke":
        sides = d.get("sides")
        return Stroke(
            color=Color.from_dict(d.get("color", {})),
            width=float(d.get("width", 1.0)),
            align=d.get("align", "inside"),
            sides=tuple(float(v) for v in sides) if sides else None,
        )


@dataclass
class Shadow:
    color: Color = field(default_factory=lambda: Color(0, 0, 0, 0.2))
    dx: float = 0.0
    dy: float = 2.0
    blur: float = 4.0
    spread: float = 0.0
    inner: bool = False

    def __post_init__(self) -> None:
        self.dx, self.dy, self.blur, self.spread = float(self.dx), float(self.dy), float(self.blur), float(self.spread)

    def to_dict(self) -> dict:
        return {"color": self.color.to_dict(), "dx": self.dx, "dy": self.dy, "blur": self.blur, "spread": self.spread, "inner": self.inner}

    @staticmethod
    def from_dict(d: dict) -> "Shadow":
        return Shadow(
            color=Color.from_dict(d.get("color", {"r": 0, "g": 0, "b": 0, "a": 0.2})),
            dx=float(d.get("dx", 0)), dy=float(d.get("dy", 0)), blur=float(d.get("blur", 0)),
            spread=float(d.get("spread", 0)), inner=bool(d.get("inner", False)),
        )


# --------------------------------------------------------------------------- text
@dataclass
class TextStyle:
    family: str = "Roboto"
    size: float = 14.0
    weight: int = 400  # 100..900
    line_height: Optional[float] = None  # px; None = normal
    letter_spacing: float = 0.0  # px
    color: Color = field(default_factory=lambda: Color(0, 0, 0))
    align: Literal["left", "center", "right"] = "left"
    valign: Literal["top", "middle", "bottom"] = "top"
    italic: bool = False
    decoration: Literal["none", "underline", "line-through"] = "none"
    transform: Literal["none", "uppercase", "lowercase", "capitalize"] = "none"

    def __post_init__(self) -> None:
        self.size = float(self.size)
        self.weight = int(self.weight)
        self.letter_spacing = float(self.letter_spacing)
        if self.line_height is not None:
            self.line_height = float(self.line_height)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["color"] = self.color.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "TextStyle":
        return TextStyle(
            family=d.get("family", "Roboto"), size=float(d.get("size", 14)), weight=int(d.get("weight", 400)),
            line_height=float(d["line_height"]) if d.get("line_height") is not None else None,
            letter_spacing=float(d.get("letter_spacing", 0)), color=Color.from_dict(d.get("color", {})),
            align=d.get("align", "left"), valign=d.get("valign", "top"), italic=bool(d.get("italic", False)),
            decoration=d.get("decoration", "none"), transform=d.get("transform", "none"),
        )


# --------------------------------------------------------------------------- layout
@dataclass
class Layout:
    """Inferred auto-layout (Figma auto-layout / CSS flex). mode='none' = absolute children."""
    mode: LayoutMode = "none"
    gap: float = 0.0
    padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # top, right, bottom, left
    align_items: Align = "start"  # cross axis
    justify: Justify = "start"  # main axis
    wrap: bool = False
    sizing_h: Sizing = "fixed"
    sizing_v: Sizing = "fixed"

    def __post_init__(self) -> None:
        self.gap = float(self.gap)
        self.padding = tuple(float(v) for v in self.padding)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["padding"] = list(self.padding)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        p = d.get("padding", [0, 0, 0, 0])
        return Layout(
            mode=d.get("mode", "none"), gap=float(d.get("gap", 0)), padding=tuple(float(v) for v in p),
            align_items=d.get("align_items", "start"), justify=d.get("justify", "start"), wrap=bool(d.get("wrap", False)),
            sizing_h=d.get("sizing_h", "fixed"), sizing_v=d.get("sizing_v", "fixed"),
        )


# --------------------------------------------------------------------------- design-system refs
@dataclass
class ComponentRef:
    """Reference to a design-system component this node was mapped to."""
    key: str  # figma component key / library id, stable
    name: str  # human name e.g. "Button"
    library: str = ""  # e.g. "material3"
    variant: dict[str, str] = field(default_factory=dict)  # e.g. {"style": "filled", "size": "md"}
    props: dict[str, object] = field(default_factory=dict)  # slot values e.g. {"label": "Save", "icon": "add"}
    confidence: float = 0.0  # 0..1
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ComponentRef":
        return ComponentRef(
            key=d.get("key", ""), name=d.get("name", ""), library=d.get("library", ""),
            variant=dict(d.get("variant", {})), props=dict(d.get("props", {})),
            confidence=float(d.get("confidence", 0)), evidence=dict(d.get("evidence", {})),
        )


# --------------------------------------------------------------------------- node / document
def new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Node:
    id: str = field(default_factory=new_id)
    type: NodeType = "frame"
    name: str = ""
    box: Box = field(default_factory=Box)
    fills: list[Fill] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)
    effects: list[Shadow] = field(default_factory=list)
    radius: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # tl, tr, br, bl
    opacity: float = 1.0
    clip: bool = False  # clips children to box
    visible: bool = True
    # text
    text: Optional[str] = None
    text_style: Optional[TextStyle] = None
    # icon / image
    icon_name: Optional[str] = None  # e.g. Material Symbol name "search"
    image_ref: Optional[str] = None
    # layout
    layout: Optional[Layout] = None
    # design-system mapping
    component: Optional[ComponentRef] = None
    tokens: dict[str, str] = field(default_factory=dict)  # prop -> token name, e.g. {"fill": "md.sys.color.primary"}
    # free-form evidence / debug
    meta: dict = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)

    def __post_init__(self) -> None:
        r = self.radius
        if isinstance(r, (int, float)):
            r = (r, r, r, r)
        self.radius = tuple(float(v) for v in r)
        self.opacity = float(self.opacity)

    # ---- convenience
    @property
    def fill_color(self) -> Optional[Color]:
        for f in self.fills:
            if f.kind == "solid" and f.color is not None:
                return f.color
        return None

    def uniform_radius(self) -> float:
        return sum(self.radius) / 4.0

    def walk(self) -> Iterator["Node"]:
        """Pre-order traversal including self."""
        yield self
        for c in self.children:
            yield from c.walk()

    def walk_with_depth(self, depth: int = 0) -> Iterator[tuple["Node", int]]:
        yield self, depth
        for c in self.children:
            yield from c.walk_with_depth(depth + 1)

    def find(self, node_id: str) -> Optional["Node"]:
        for n in self.walk():
            if n.id == node_id:
                return n
        return None

    def parent_of(self, node_id: str) -> Optional["Node"]:
        for n in self.walk():
            for c in n.children:
                if c.id == node_id:
                    return n
        return None

    def leaves(self) -> list["Node"]:
        return [n for n in self.walk() if not n.children]

    def count(self) -> int:
        return sum(1 for _ in self.walk())

    def is_leaf(self) -> bool:
        return not self.children

    def fit_to_children(self, pad: float = 0.0) -> None:
        if not self.children:
            return
        b = self.children[0].box
        for c in self.children[1:]:
            b = b.union(c.box)
        self.box = b.expand(pad)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "box": self.box.to_dict(),
            "fills": [f.to_dict() for f in self.fills],
            "strokes": [s.to_dict() for s in self.strokes],
            "effects": [e.to_dict() for e in self.effects],
            "radius": list(self.radius),
            "opacity": self.opacity,
            "clip": self.clip,
            "visible": self.visible,
            "text": self.text,
            "text_style": self.text_style.to_dict() if self.text_style else None,
            "icon_name": self.icon_name,
            "image_ref": self.image_ref,
            "layout": self.layout.to_dict() if self.layout else None,
            "component": self.component.to_dict() if self.component else None,
            "tokens": dict(self.tokens),
            "meta": self.meta,
            "children": [c.to_dict() for c in self.children],
        }
        return d

    @staticmethod
    def from_dict(d: dict) -> "Node":
        r = d.get("radius", [0, 0, 0, 0])
        if isinstance(r, (int, float)):
            r = [r, r, r, r]
        return Node(
            id=d.get("id") or new_id(),
            type=d.get("type", "frame"),
            name=d.get("name", ""),
            box=Box.from_dict(d.get("box", {})),
            fills=[Fill.from_dict(f) for f in d.get("fills", [])],
            strokes=[Stroke.from_dict(s) for s in d.get("strokes", [])],
            effects=[Shadow.from_dict(e) for e in d.get("effects", [])],
            radius=tuple(float(v) for v in r),
            opacity=float(d.get("opacity", 1.0)),
            clip=bool(d.get("clip", False)),
            visible=bool(d.get("visible", True)),
            text=d.get("text"),
            text_style=TextStyle.from_dict(d["text_style"]) if d.get("text_style") else None,
            icon_name=d.get("icon_name"),
            image_ref=d.get("image_ref"),
            layout=Layout.from_dict(d["layout"]) if d.get("layout") else None,
            component=ComponentRef.from_dict(d["component"]) if d.get("component") else None,
            tokens=dict(d.get("tokens", {})),
            meta=dict(d.get("meta", {})),
            children=[Node.from_dict(c) for c in d.get("children", [])],
        )


@dataclass
class Document:
    width: int
    height: int
    root: Node
    dpr: float = 1.0
    source_image: Optional[str] = None  # path to the reference screenshot
    palette: list[Color] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    design_system: Optional[str] = None  # name of DS used for mapping
    meta: dict = field(default_factory=dict)
    version: str = "1"

    # ---- helpers
    def walk(self) -> Iterator[Node]:
        return self.root.walk()

    def find(self, node_id: str) -> Optional[Node]:
        return self.root.find(node_id)

    def parent_of(self, node_id: str) -> Optional[Node]:
        return self.root.parent_of(node_id)

    def texts(self) -> list[Node]:
        return [n for n in self.walk() if n.type == "text" and n.text]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "dpr": self.dpr,
            "source_image": self.source_image,
            "palette": [c.to_dict() for c in self.palette],
            "fonts": list(self.fonts),
            "design_system": self.design_system,
            "meta": self.meta,
            "root": self.root.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(
            width=int(d["width"]), height=int(d["height"]), root=Node.from_dict(d["root"]),
            dpr=float(d.get("dpr", 1.0)), source_image=d.get("source_image"),
            palette=[Color.from_dict(c) for c in d.get("palette", [])], fonts=list(d.get("fonts", [])),
            design_system=d.get("design_system"), meta=dict(d.get("meta", {})), version=str(d.get("version", "1")),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> "Document":
        return Document.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "Document":
        with open(path) as f:
            return Document.from_json(f.read())

    @staticmethod
    def blank(width: int, height: int, bg: Color | str = "#ffffff") -> "Document":
        root = Node(id="root", type="frame", name="Screen", box=Box(0, 0, width, height), fills=[Fill.solid(bg)], clip=True)
        return Document(width=width, height=height, root=root)


# --------------------------------------------------------------------------- utilities
def assign_ids(root: Node, prefix: str = "n") -> None:
    """Deterministic ids in pre-order (useful for tests / diffs)."""
    for i, n in enumerate(root.walk()):
        n.id = "root" if i == 0 else f"{prefix}{i}"


def flatten(root: Node) -> list[Node]:
    return list(root.walk())


def depth_of(root: Node, node_id: str) -> int:
    for n, d in root.walk_with_depth():
        if n.id == node_id:
            return d
    return -1


def summarize(doc: Document, max_nodes: int = 60) -> str:
    """Human-readable tree dump for logs/debugging."""
    lines = [f"Document {doc.width}x{doc.height} dpr={doc.dpr} nodes={doc.root.count()}"]
    for i, (n, d) in enumerate(doc.root.walk_with_depth()):
        if i >= max_nodes:
            lines.append("  ...")
            break
        b = n.box
        extra = ""
        if n.type == "text":
            extra = f' "{(n.text or "")[:30]}" {n.text_style.size if n.text_style else "?"}px/{n.text_style.weight if n.text_style else "?"}'
        elif n.fill_color is not None:
            extra = f" fill={n.fill_color.hex()}"
        if n.component:
            extra += f" -> {n.component.name}{n.component.variant}"
        lines.append(f"{'  ' * d}{n.type}:{n.id} [{b.x:.0f},{b.y:.0f} {b.w:.0f}x{b.h:.0f}]{extra}")
    return "\n".join(lines)
