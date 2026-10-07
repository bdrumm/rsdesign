"""`dt doctor`: verify that this checkout can run the pipeline, and say exactly how to fix what is missing.

Each check returns a ``Check(name, status, detail, fix)`` with status ``ok`` | ``warn`` | ``fail``.
``fail`` means a core stage cannot run (exit code 1); ``warn`` means a degraded or optional feature
(e.g. renders on Playwright's Chromium instead of the Chrome the baselines were made with, or the
git-ignored reference screenshots are absent). Every check runs the real thing (launches the
browser, renders text with the self-hosted fonts, OCRs it) rather than probing imports only.
"""
from __future__ import annotations

import glob
import importlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_MAC = sys.platform == "darwin"
PIP = "uv pip install" if shutil.which("uv") else "python -m pip install"


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


# --------------------------------------------------------------------------- checks
def check_python() -> Check:
    v = sys.version_info
    if v < (3, 12):
        return Check("python", "fail", f"Python {v.major}.{v.minor} (< 3.12)", "create the venv with Python >= 3.12: uv venv --python 3.12 .venv")
    return Check("python", "ok", f"Python {v.major}.{v.minor}.{v.micro} at {sys.executable}")


def check_packages() -> Check:
    mods = {"numpy": "numpy", "cv2": "opencv-python-headless", "PIL": "pillow", "skimage": "scikit-image",
            "scipy": "scipy", "playwright": "playwright", "requests": "requests"}
    missing, broken = [], []
    for mod, dist in mods.items():
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            missing.append(dist)
        except Exception as e:  # e.g. cv2 without libGL on a headless Linux box
            broken.append(f"{mod}: {str(e).splitlines()[0]}")
    if missing:
        return Check("python packages", "fail", "missing: " + ", ".join(missing), f"{PIP} -e .")
    if broken:
        fix = f"{PIP} --reinstall opencv-python-headless" if any(b.startswith("cv2") for b in broken) else f"{PIP} -e ."
        return Check("python packages", "fail", "; ".join(broken), fix)
    try:
        both = md.version("opencv-python") and md.version("opencv-python-headless")
    except md.PackageNotFoundError:
        both = None
    if both:
        # rapidocr pulls in the GUI build; both write cv2/. Harmless on macOS, needs libGL on Linux.
        return Check("python packages", "warn" if not IS_MAC else "ok",
                     "opencv-python and opencv-python-headless are both installed (rapidocr pulls the GUI build); they share cv2/",
                     "" if IS_MAC else f"{PIP} --reinstall opencv-python-headless  (keeps cv2 working without libGL)")
    return Check("python packages", "ok", "numpy, opencv, pillow, scikit-image, scipy, playwright, requests")


def check_node() -> Check:
    node = shutil.which("node")
    if not node:
        return Check("node", "warn", "node not found: corpus builds (mwc) and the headless Figma plugin tests need it",
                     "install Node >= 18 (https://nodejs.org or your package manager), then: npm ci")
    v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    ok_mods = os.path.isdir(_p("node_modules", "@material", "web")) and os.path.isdir(_p("node_modules", "esbuild"))
    if not ok_mods:
        return Check("node", "warn", f"node {v}; node_modules incomplete (@material/web, esbuild)", "npm ci")
    return Check("node", "ok", f"node {v}; node_modules present")


def check_bundle() -> Check:
    b = _p("fixtures", "mwc", "mwc.bundle.js")
    if not os.path.exists(b) or os.path.getsize(b) < 100_000:
        return Check("mwc bundle", "fail", "fixtures/mwc/mwc.bundle.js missing or truncated", "npm ci && npm run build:mwc")
    return Check("mwc bundle", "ok", f"fixtures/mwc/mwc.bundle.js ({os.path.getsize(b) // 1024} KB)")


def check_fonts() -> Check:
    css = sorted(glob.glob(_p("fixtures", "fonts", "*.local.css")))
    if not css:
        return Check("fonts", "fail", "no fixtures/fonts/*.local.css", "restore fixtures/fonts from git: git checkout -- fixtures/fonts")
    missing = []
    for c in css:
        for u in re.findall(r"url\(['\"]?([^'\")]+)", open(c).read()):
            if not u.startswith(("http:", "https:", "data:", "file:")) and not os.path.exists(os.path.join(os.path.dirname(c), u)):
                missing.append(u)
    if missing:
        return Check("fonts", "fail", f"{len(missing)} font files referenced by fixtures/fonts/*.local.css are missing (e.g. {missing[0]})",
                     "git checkout -- fixtures/fonts")
    return Check("fonts", "ok", f"{len(css)} self-hosted font stylesheets, all referenced files present")


def check_browser() -> Check:
    try:
        from dt.render import screenshot as s
        import numpy as np

        img = s.html_to_png("<style>html,body{margin:0;background:#000}</style><div style='margin-left:30px;width:30px;height:30px;background:#fff'></div>", 60, 30)
        if img.shape[:2] != (30, 60) or int(img[:, :28].max()) > 10 or int(img[:, 32:].min()) < 245:
            return Check("browser", "fail", "browser started but the test render is wrong", "reinstall the browser: python -m playwright install chromium")
        used = s.BROWSER_USED or "?"
        if used != "chrome":
            return Check("browser", "warn", f"rendering with {used} (system Chrome unavailable); text antialiasing can differ from the "
                         "Chrome-made baselines (knowledge/baseline.json)", "install Google Chrome for baseline-comparable renders")
        return Check("browser", "ok", "system Chrome via Playwright (channel=chrome)")
    except Exception as e:
        from dt.render.screenshot import BROWSER_HELP
        return Check("browser", "fail", str(e).splitlines()[0][:300], BROWSER_HELP)


def _render_text(text: str, family: str, size: int, w: int, h: int):
    from dt.render.html import _font_face_css
    from dt.render.screenshot import html_to_png

    html = (f"<style>{_font_face_css()} body{{margin:0;background:#fff}}</style>"
            f"<div style=\"position:absolute;left:8px;top:6px;font:{size}px '{family}', monospace;color:#1d1b20;white-space:nowrap\">{text}</div>")
    return html_to_png(html, w, h)


def check_font_render() -> Check:
    """Roboto must actually load: a missing font silently falls back to monospace (wider ink)."""
    try:
        import numpy as np

        def ink_w(img) -> int:
            cols = np.flatnonzero((img.min(axis=2) < 128).any(axis=0))
            return int(cols[-1] - cols[0] + 1) if cols.size else 0
        t = "Hamburgefonstiv"
        r = ink_w(_render_text(t, "Roboto", 20, 300, 40))
        m = ink_w(_render_text(t, "monospace", 20, 300, 40))
        if r == 0 or abs(r - m) < 3:
            return Check("font render", "fail", f"Roboto did not load (ink width {r}px == monospace fallback {m}px)",
                         "git checkout -- fixtures/fonts ; check that file:// URLs are readable")
        return Check("font render", "ok", f"Roboto renders ({r}px ink vs {m}px monospace fallback)")
    except Exception as e:
        return Check("font render", "fail", str(e).splitlines()[0][:300], "fix the browser check first")


def check_ocr() -> Check:
    try:
        from dt.perceive import ocr
    except Exception as e:
        return Check("ocr", "fail", f"dt.perceive.ocr import failed: {e}", f"{PIP} -e .")
    names = ocr.available_backends()
    fix = (f"{PIP} -e '.[ocr]'  (RapidOCR, pure pip) or install tesseract + pytesseract"
           if not IS_MAC else f"{PIP} -e .  (installs pyobjc Vision)")
    if not names:
        return Check("ocr", "fail", "no OCR backend: perceive cannot read text", fix)
    try:
        img = _render_text("Doctor check 42", "Roboto", 20, 260, 40)
    except Exception:  # browser broken (reported by its own check): test OCR on a PIL-drawn line instead
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        im = Image.new("RGB", (260, 40), "white")
        ImageDraw.Draw(im).text((8, 8), "Doctor check 42", fill=(29, 27, 32), font=ImageFont.load_default(size=20))
        img = np.asarray(im)
    try:
        be = ocr.get_backend()
        got = " ".join(l.text for l in ocr.ocr_lines(img, be))
    except Exception as e:
        return Check("ocr", "fail", f"{names}: OCR run failed: {str(e).splitlines()[0][:200]}", fix)
    want = os.environ.get("DT_OCR") or "auto"
    detail = f"backend {be.name} (available {names}, DT_OCR={want}); read {got!r}"
    if got.replace(" ", "").lower() != "doctorcheck42":
        return Check("ocr", "warn", detail + " — expected 'Doctor check 42'", "try another backend: DT_OCR=vision|rapidocr|tesseract")
    if be.name != "vision" and IS_MAC:
        return Check("ocr", "warn", detail + "; the shipped baselines were made with Vision", "unset DT_OCR to use Vision")
    return Check("ocr", "ok", detail)


def check_icon_atlas() -> Check:
    paths = [_p("fixtures", "icons", f"atlas_fill{f}_32.npz") for f in (0, 1)]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        return Check("icon atlas", "warn", f"missing {missing}: built on first use (slow, needs the browser)",
                     "python -c \"from dt.perceive.icons import load_atlas; load_atlas()\"")
    try:
        import numpy as np
        n = sum(len(np.load(p)["names"]) for p in paths)
    except Exception as e:
        return Check("icon atlas", "fail", f"atlas unreadable: {e}", "delete fixtures/icons/atlas_*.npz and rebuild: python -c \"from dt.perceive.icons import load_atlas; load_atlas()\"")
    return Check("icon atlas", "ok", f"{n} glyph templates")


def check_reference_page() -> Check:
    png, html = _p("out", "mwc_test.png"), _p("out", "mwc_test.html")
    if not (os.path.exists(png) and os.path.exists(html)):
        return Check("reference page", "warn", "out/mwc_test.{html,png} missing: tests that use the Material Web reference page are skipped",
                     "python scripts/make_reference_page.py")
    if _p("fixtures") not in open(html).read():
        return Check("reference page", "warn", "out/mwc_test.html points at another checkout", "python scripts/make_reference_page.py --force")
    return Check("reference page", "ok", "out/mwc_test.html + .png")


def check_reference_screens() -> Check:
    n = len(glob.glob(_p("fixtures", "screens", "*.png")))
    if n < 15:
        return Check("reference screens", "warn", f"{n} PNGs in fixtures/screens (git-ignored, optional): real-screen fidelity checks are skipped",
                     "dt corpus build --kind screens   (network)")
    return Check("reference screens", "ok", f"{n} PNGs in fixtures/screens")


CHECKS: list[Callable[[], Check]] = [check_python, check_packages, check_node, check_bundle, check_fonts, check_browser,
                                     check_font_render, check_ocr, check_icon_atlas, check_reference_page, check_reference_screens]


def run(checks: Optional[list[Callable[[], Check]]] = None) -> list[Check]:
    out = []
    for c in checks or CHECKS:
        try:
            out.append(c())
        except Exception as e:  # a check must never crash the doctor
            out.append(Check(c.__name__.replace("check_", ""), "fail", f"check crashed: {type(e).__name__}: {e}"))
    return out


def summary(results: list[Check]) -> dict:
    return {"ok": not any(r.status == "fail" for r in results), "platform": sys.platform,
            "checks": [asdict(r) for r in results]}


def format_human(results: list[Check]) -> str:
    tag = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    lines = []
    for r in results:
        lines.append(f"{tag.get(r.status, r.status)} {r.name:18s} {r.detail}")
        if r.fix and r.status != "ok":
            lines.append(f"       {'':18s} fix: {r.fix}")
    nf = sum(r.status == "fail" for r in results)
    nw = sum(r.status == "warn" for r in results)
    lines.append(f"\n{'READY' if not nf else 'NOT READY'}: {nf} failing, {nw} warnings ({sys.platform})")
    return "\n".join(lines)
