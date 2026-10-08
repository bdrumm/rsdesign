"""No browser-side wait may be unbounded: a page Chrome considers hidden pauses requestAnimationFrame, and an
unbounded await then hangs the whole process silently (seen once as a frozen scenario gate). Every
animation-frame or font wait must race a timer (Promise.race with setTimeout, or the cap() helper)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = [re.compile(r"await\s+new Promise\(r\s*=>\s*requestAnimationFrame"),
            re.compile(r"await\s+document\.fonts\.ready\s*;"),
            re.compile(r"\.then\(\(\)\s*=>\s*new Promise\(r\s*=>\s*requestAnimationFrame"),
            re.compile(r'evaluate\("document\.fonts\.ready')]


def test_no_unbounded_browser_waits():
    bad = []
    for dp, _, fs in os.walk(os.path.join(ROOT, "dt")):
        for fn in fs:
            if fn.endswith(".py"):
                path = os.path.join(dp, fn)
                for i, line in enumerate(open(path), 1):
                    if any(p.search(line) for p in PATTERNS):
                        bad.append(f"{os.path.relpath(path, ROOT)}:{i}: {line.strip()[:100]}")
    assert not bad, "unbounded browser waits:\n" + "\n".join(bad)
