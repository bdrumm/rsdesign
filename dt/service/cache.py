"""Content-addressed render cache and the fingerprints that make it safe.

A cached render is keyed by ``sha256(html bytes, width, height, dpr, render options, renderer
fingerprint)``. The renderer fingerprint covers everything else that decides the pixels:

* the browser (name + version, reported by the worker that rendered it),
* the font files the page can load: ``fixtures/fonts`` and the font files in ``fixtures/icons`` of
  the checkout whose absolute paths the HTML carries (each client sends its repo root), plus
  ``dt/render/html.py`` and ``dt/render/screenshot.py`` (the page-load / font-wait protocol),
* the size + mtime of any other ``file://`` resource the HTML references (images, scripts).

Fingerprints hash file contents, memoised on ``(path, size, mtime_ns)`` so a re-check costs one
``stat`` per file (and is itself trusted for ``_TTL_S`` seconds).

:class:`RenderCache` stores ``<DT_HOME>/cache/render/<k[:2]>/<k>.png`` with an LRU size cap
(``service.cache.max_mb``; eviction trims to 90 % of the cap, oldest use first; a hit refreshes the
file's mtime so the order survives restarts).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Iterable, Optional

_TTL_S = 1.0
_FONT_EXT = (".woff2", ".woff", ".ttf", ".otf", ".css")
_memo_lock = threading.Lock()
_content_memo: dict[str, tuple[tuple, str]] = {}   # path -> ((size, mtime_ns), sha)
_set_memo: dict[tuple, tuple[float, str]] = {}      # (kind, root) -> (checked_at, sha)


def _file_sha(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_size, st.st_mtime_ns)
    with _memo_lock:
        hit = _content_memo.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with _memo_lock:
        _content_memo[path] = (sig, sha)
    return sha


def files_digest(paths: Iterable[str], root: str) -> str:
    """sha256 over (relative path, content sha) of `paths` (missing files count as absent)."""
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        s = _file_sha(p)
        if s is not None:
            h.update(os.path.relpath(p, root).encode() + b"\0" + s.encode() + b"\n")
    return h.hexdigest()


def _memoised(kind: str, root: str, compute) -> str:
    key = (kind, os.path.abspath(root))
    now = time.monotonic()
    with _memo_lock:
        hit = _set_memo.get(key)
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    sha = compute()
    with _memo_lock:
        _set_memo[key] = (now, sha)
    return sha


def asset_files(root: str) -> list[str]:
    fonts = [p for p in glob.glob(os.path.join(root, "fixtures", "fonts", "*")) if p.endswith(_FONT_EXT)]
    icons = [p for p in glob.glob(os.path.join(root, "fixtures", "icons", "*")) if p.endswith(_FONT_EXT)]
    src = [os.path.join(root, "dt", "render", "html.py"), os.path.join(root, "dt", "render", "screenshot.py")]
    return fonts + icons + src


def asset_fingerprint(root: str) -> str:
    """Fonts (fixtures/fonts, icon font files) + renderer sources of the checkout at `root`."""
    return _memoised("assets", root, lambda: files_digest(asset_files(root), root))


def code_files(root: str) -> list[str]:
    pats = [os.path.join(root, "dt", "**", "*.py"), os.path.join(root, "dt", "params.json"),
            os.path.join(root, "fixtures", "design_systems", "*"), os.path.join(root, "fixtures", "icons", "*"),
            os.path.join(root, "fixtures", "fonts", "*")]
    out: list[str] = []
    for p in pats:
        out.extend(f for f in glob.glob(p, recursive=True) if os.path.isfile(f) and "__pycache__" not in f)
    return out


def code_fingerprint(root: str) -> str:
    """Everything a perceive / bench job's result depends on in the checkout at `root` (all of dt/,
    dt/params.json, design systems, fonts, icon atlases). Jobs that run pipeline code are only served
    when the client's code fingerprint equals the service's."""
    return _memoised("code", root, lambda: files_digest(code_files(root), root))


_FILE_URL = re.compile(r"file://([^'\")\s<>]+)")


def referenced_files_sig(html: str, skip_dirs: Iterable[str] = ()) -> str:
    """(path, size, mtime_ns) of every file:// resource in `html` outside `skip_dirs` (fonts are
    covered by :func:`asset_fingerprint`)."""
    skip = tuple(os.path.abspath(d) + os.sep for d in skip_dirs)
    h = hashlib.sha256()
    for p in sorted(set(_FILE_URL.findall(html))):
        if skip and p.startswith(skip):
            continue
        try:
            st = os.stat(p)
            h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{p}\0missing\n".encode())
    return h.hexdigest()


def render_key(html: str, width: int, height: int, dpr: float, opts: dict, fingerprint: str) -> str:
    h = hashlib.sha256()
    h.update(html.encode("utf-8"))
    h.update(b"\0" + json.dumps({"w": int(width), "h": int(height), "dpr": float(dpr), "opts": opts, "fp": fingerprint},
                                sort_keys=True).encode())
    return h.hexdigest()


class RenderCache:
    """Disk LRU of PNG bytes by key. Thread-safe within one process."""

    def __init__(self, root: str, max_bytes: int):
        self.root = root
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._index: "OrderedDict[str, int]" = OrderedDict()  # key -> size, least recently used first
        self.bytes = 0
        self.hits = self.misses = self.stores = self.evictions = 0
        self.saved_s = 0.0  # render time the hits did not spend (known for entries stored by this process)
        self._cost: dict[str, float] = {}
        os.makedirs(root, exist_ok=True)
        found = []
        for p in glob.glob(os.path.join(root, "??", "*.png")):
            try:
                st = os.stat(p)
            except OSError:
                continue
            found.append((st.st_mtime, os.path.basename(p)[:-4], st.st_size))
        for _m, k, size in sorted(found):
            self._index[k] = size
            self.bytes += size
        self._evict()

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".png")

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            known = key in self._index
        data = None
        if known:
            try:
                with open(self._path(key), "rb") as f:
                    data = f.read()
                os.utime(self._path(key))
            except OSError:
                data = None
        with self._lock:
            if data is None:
                if known and key in self._index:  # deleted behind our back
                    self.bytes -= self._index.pop(key)
                self.misses += 1
            else:
                self._index.move_to_end(key)
                self.hits += 1
                self.saved_s += self._cost.get(key, 0.0)
        return data

    def put(self, key: str, data: bytes, cost_s: float = 0.0) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        with self._lock:
            if key in self._index:
                self.bytes -= self._index.pop(key)
            self._index[key] = len(data)
            self.bytes += len(data)
            self.stores += 1
            if cost_s:
                self._cost[key] = float(cost_s)
            self._evict()

    def _evict(self) -> None:
        if self.bytes <= self.max_bytes:
            return
        target = int(self.max_bytes * 0.9)
        while self._index and self.bytes > target:
            key, size = self._index.popitem(last=False)
            self.bytes -= size
            self.evictions += 1
            self._cost.pop(key, None)
            try:
                os.remove(self._path(key))
            except OSError:
                pass

    def clear(self) -> int:
        with self._lock:
            n = len(self._index)
            for key in list(self._index):
                try:
                    os.remove(self._path(key))
                except OSError:
                    pass
            self._index.clear()
            self._cost.clear()
            self.bytes = 0
            return n

    def stats(self) -> dict:
        with self._lock:
            looked = self.hits + self.misses
            return {"dir": self.root, "entries": len(self._index), "bytes": self.bytes, "max_bytes": self.max_bytes,
                    "hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / looked, 4) if looked else None,
                    "stores": self.stores, "evictions": self.evictions, "saved_s": round(self.saved_s, 3)}
