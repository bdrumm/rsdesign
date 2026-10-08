"""User-informed tuning: learn from a consumer's feedback bundles, measure it on *their* screens, keep it only
when it helps them without hurting the shared bench.

* :func:`build_corpus`  -- ``$DT_HOME/feedback/corpus/<id>/`` (``<id>.png`` + corrected ``<id>.gt.json`` +
  ``manifest.json``): one bench corpus per bundle, so ``dt bench --corpus $DT_HOME/feedback/corpus/*`` works.
* :func:`observations` / :func:`derive_rules` -- component corrections (and "ok" confirmations of a matched
  component) grouped by the matcher's learned signature; votes are counted per distinct *screenshot*, so one
  screen with six identical chips is one piece of evidence, not six; conflicting answers lower confidence; a
  rule activates at ``support >= feedback.rule_min_support`` and ``confidence >= feedback.rule_min_conf``.
* :func:`learn` -- derive rules -> evaluate the user corpus before / after (item accuracy + structural composite)
  -> global bench gate -> keep or discard -> one ledger entry (scope local) with the exact diff and evidence.
* :func:`evaluate` / :func:`eval_history` -- re-run every feedback case with the current model and append to
  ``$DT_HOME/feedback/history.jsonl``: accuracy on the consumer's own use cases over model versions.
* :func:`revert` -- restore the rules a ledger entry replaced (``kind: revert``).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Callable, Iterable, Optional

from dt.feedback import ledger
from dt.feedback.store import (corpus_root, dt_home, feedback_root, history_path, list_bundles, load_bundle,
                               load_target, local_rules_path, model_version, now_iso, sha256_file)
from dt.ir import Document, Node
from dt.params import P, register

register("feedback.rule_min_support", 2,
         "distinct screenshots whose corrections must agree before a learned matcher rule activates", (1, 10))
register("feedback.rule_min_conf", 0.67,
         "share of agreeing screenshots, support / (support + against), a learned rule needs to activate", (0.5, 1.0))
register("feedback.global_min_support", 3,
         "distinct contributed screenshots a candidate global rule needs before its promotion is even tried", (1, 20))
register("feedback.eval.iou", 0.5, "IoU at which a predicted node answers a feedback item's box", (0.1, 0.95))
register("feedback.eval.geom_iou", 0.85,
         "IoU between a predicted node and a corrected box at which a geometry correction counts as reproduced", (0.5, 0.99))
register("feedback.gate.user_eps", 0.0,
         "largest tolerated drop of user-corpus item accuracy / composite when accepting learned rules", (0.0, 0.05))

Logger = Optional[Callable[[str], None]]
RULES_SCHEMA = "rsdesign.learned-rules/1"
CONTAINER_TYPES = ("frame", "rect", "ellipse", "vector", "instance")


def _say(log: Logger) -> Callable[[str], None]:
    return log or (lambda _m: None)


# --------------------------------------------------------------------------- user corpus
def bundle_target_path(bundle: dict, home: Optional[str] = None) -> Optional[str]:
    """A PNG of the bundle's screenshot at IR resolution: the stored copy (consent.store_screenshot), else the
    original file when it is unchanged and already at dpr 1 (never copied)."""
    d = os.path.join(feedback_root(home), bundle["id"])
    t = os.path.join(d, "target.png")
    if os.path.exists(t):
        return t
    run = bundle.get("run") or {}
    src = run.get("source_path")
    if src and os.path.exists(src) and float(run.get("dpr") or 1.0) == 1.0 and src.lower().endswith(".png") \
            and run.get("screenshot_sha256") == sha256_file(src):
        return src
    return None


def build_corpus(home: Optional[str] = None, bundles: Optional[list[dict]] = None) -> list[str]:
    """(Re)write the user corpus; returns one corpus dir per usable bundle (others are skipped: no screenshot)."""
    root = corpus_root(home)
    os.makedirs(root, exist_ok=True)
    bundles = list_bundles(home) if bundles is None else bundles
    dirs: list[str] = []
    keep = set()
    for b in bundles:
        png = bundle_target_path(b, home)
        ir = os.path.join(feedback_root(home), b["id"], "ir.json")
        if png is None or not os.path.exists(ir):
            continue
        d = os.path.join(root, b["id"])
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, b["id"] + ".png")
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.dirname(os.path.abspath(png)) == os.path.join(feedback_root(home), b["id"]):
            shutil.copyfile(png, dst)
        else:
            os.symlink(os.path.abspath(png), dst)  # the user's own file, not copied without consent
        gt = Document.load(ir)
        gt.source_image = dst
        gt.save(os.path.join(d, b["id"] + ".gt.json"))
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"kind": "user_feedback", "n": 1, "ids": [b["id"]],
                       "cases": [{"id": b["id"], "bundle": b["id"], "width": gt.width, "height": gt.height,
                                  "items": len(b.get("items") or []), "model": (b.get("run") or {}).get("model", {}).get("id")}]},
                      f, indent=2)
        dirs.append(d)
        keep.add(b["id"])
    for name in os.listdir(root):  # bundles deleted since: drop their cases
        if name.startswith("fb-") and name not in keep:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    return dirs


def corpus_dirs(home: Optional[str] = None) -> list[str]:
    root = corpus_root(home)
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, n) for n in sorted(os.listdir(root))
            if n.startswith("fb-") and os.path.exists(os.path.join(root, n, n + ".gt.json"))]


# --------------------------------------------------------------------------- rules
def _variant_key(v: dict) -> tuple:
    return tuple(sorted((str(k).lower(), str(x).lower()) for k, x in (v or {}).items()))


def _normalize_variant(name: Optional[str], variant: dict, ds_name: Optional[str]) -> dict:
    """Project an answer's variant onto the design system's vocabulary (so a partial answer such as
    ``{"type": "filter"}`` and a full one with the default ``selected=false`` are the same answer)."""
    if not name or not variant:
        return dict(variant or {})
    from dt.feedback.capture import design_system_for
    ds = design_system_for(ds_name)
    spec = next((s for s in ds.components if s.name.lower() == str(name).lower()), None)
    return ds.normalize_variant(spec.name, dict(variant)) if spec is not None else {str(k): str(v) for k, v in variant.items()}


def observations(bundles: Iterable[dict]) -> list[dict]:
    """One observation per component answer on a container node with a learned signature."""
    out = []
    for b in bundles:
        ds_name = (b.get("run") or {}).get("design_system")
        screen = (b.get("run") or {}).get("screenshot_sha256") or b["id"]
        for it in b.get("items") or []:
            ctx = it.get("context") or {}
            if not ctx.get("signature_hash") or ctx.get("type") not in CONTAINER_TYPES:
                continue
            if it["kind"] == "component":
                name, variant = (it.get("value") or {}).get("name"), (it.get("value") or {}).get("variant") or {}
            elif it["kind"] == "ok" and ctx.get("component"):
                name, variant = ctx["component"]["name"], ctx["component"].get("variant") or {}
            else:
                continue
            out.append({"signature_hash": ctx["signature_hash"], "features": ctx.get("features") or {},
                        "component": name, "variant": _normalize_variant(name, variant, ds_name), "screen": screen,
                        "bundle": b["id"], "item": it["id"]})
    return out


def _clusters(obs: list[dict]) -> list[list[dict]]:
    """Split one signature's observations into height clusters (a 32 px chip and a 56 px card of the same
    categorical shape are different rules)."""
    tol = 2.0 * float(P["map.learned.h_tol"])
    out: list[list[dict]] = []
    for o in sorted(obs, key=lambda o: float(o["features"].get("h", 0.0))):
        h = float(o["features"].get("h", 0.0))
        if out and h - float(out[-1][-1]["features"].get("h", 0.0)) <= tol:
            out[-1].append(o)
        else:
            out.append([o])
    return out


def derive_rules(obs: list[dict], scope: str = "local", min_support: Optional[int] = None,
                 min_conf: Optional[float] = None) -> list[dict]:
    """Rules from observations (see module docstring). Deterministic for a given set of observations."""
    min_support = int(P["feedback.rule_min_support"] if min_support is None else min_support)
    min_conf = float(P["feedback.rule_min_conf"] if min_conf is None else min_conf)
    by_sig: dict[str, list[dict]] = {}
    for o in obs:
        by_sig.setdefault(o["signature_hash"], []).append(o)
    rules: list[dict] = []
    for sig, group in sorted(by_sig.items()):
        for cl in _clusters(group):
            votes: dict[tuple, set] = {}
            for o in cl:
                key = ((o["component"] or "").lower() or None, _variant_key(o["variant"]))
                votes.setdefault(key, set()).add(o["screen"])
            # the answer most screens agree on; ties -> fewer variant constraints, then alphabetical
            win = max(votes, key=lambda k: (len(votes[k]), -len(k[1]), str(k)))
            support_screens = votes[win]
            # a screen that answered both ways shows the signature is ambiguous there: it counts on both sides
            against_screens = set().union(*[s for k, s in votes.items() if k != win])
            # for a name-only rule another variant of the same component is not evidence against it;
            # for a variant rule ("these are *filter* chips") it is
            name_against = set().union(*[s for k, s in votes.items() if k[0] != win[0]])
            support, against = len(support_screens), len(against_screens if win[1] else name_against)
            conf = support / max(1, support + against)
            winners = [o for o in cl if ((o["component"] or "").lower() or None, _variant_key(o["variant"])) == win]
            name = next((o["component"] for o in winners), None)
            hs = [float(o["features"].get("h", 0.0)) for o in cl]
            asp = [float(o["features"].get("aspect", 0.0)) for o in cl]
            cat = {k: v for k, v in cl[0]["features"].items() if k not in ("h", "w", "aspect")}
            rid = "r" + hashlib.sha1(f"{sig}:{min(hs):.0f}:{win}".encode()).hexdigest()[:10]
            rules.append({
                "id": rid, "signature_hash": sig,
                "features": {**cat, "h_range": [round(min(hs), 1), round(max(hs), 1)],
                             "aspect_range": [round(min(asp), 3), round(max(asp), 3)]},
                "component": name, "variant": dict(winners[0]["variant"]) if winners else {},
                "support": support, "against": against,
                "confidence": round(conf, 3), "items": len(cl),
                "active": bool(support >= min_support and conf >= min_conf),
                "scope": scope, "sources": sorted({o["bundle"] for o in cl}),
            })
    return rules


def read_rules(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return list(json.load(f).get("rules", []))
    except (OSError, ValueError):
        return []


def write_rules(path: str, rules: list[dict], scope: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"schema": RULES_SCHEMA, "scope": scope, "updated": now_iso(), "rules": rules}, f, indent=2)
    os.replace(tmp, path)
    return path


_TRACKED = ("component", "variant", "active", "support", "against", "confidence")


def rules_diff(old: list[dict], new: list[dict]) -> dict:
    """Exact change between two rule sets: added / removed rules and, per kept rule id, field -> [old, new]."""
    o, n = {r["id"]: r for r in old}, {r["id"]: r for r in new}
    summary = lambda r: {k: r.get(k) for k in ("id", "signature_hash", "component", "variant", "support", "against", "confidence", "active")}
    changed = {}
    for rid in sorted(set(o) & set(n)):
        d = {k: [o[rid].get(k), n[rid].get(k)] for k in _TRACKED if o[rid].get(k) != n[rid].get(k)}
        if d:
            changed[rid] = d
    return {"added": [summary(n[r]) for r in sorted(set(n) - set(o))], "removed": [summary(o[r]) for r in sorted(set(o) - set(n))],
            "changed": changed}


def _diff_empty(d: dict) -> bool:
    return not (d["added"] or d["removed"] or d["changed"])


# --------------------------------------------------------------------------- evaluation on the user corpus
_PERCEIVED: dict[tuple, dict] = {}


def _perceived(png: str) -> Document:
    key = (os.path.realpath(png), os.path.getmtime(png))
    if key not in _PERCEIVED:
        from dt.perceive import perceive
        _PERCEIVED[key] = perceive(png).to_dict()
    return Document.from_dict(_PERCEIVED[key])


def _best(pred: Document, box: list[float], types: Optional[tuple] = None) -> tuple[Optional[Node], float]:
    from dt.ir import Box
    b = Box(*box)
    best, best_iou = None, 0.0
    for n in pred.walk():
        if n is pred.root or not n.visible:
            continue
        t = n.meta.get("orig_type", n.type) if n.type == "instance" else n.type
        if types is not None and t not in types and n.type not in types:
            continue
        iou = n.box.iou(b)
        if iou > best_iou:
            best, best_iou = n, iou
    return best, best_iou


def check_item(item: dict, pred: Document) -> Optional[bool]:
    """Does the model's output ``pred`` now agree with this correction? None = not checkable."""
    from dt.compare.structural import normalize_text
    thr = float(P["feedback.eval.iou"])
    kind, v, ctx = item["kind"], item.get("value") or {}, item.get("context") or {}
    box = item.get("box") or ctx.get("box")
    if box is None:
        return None
    if kind in ("component", "ok") and ctx.get("type") in CONTAINER_TYPES:
        n, iou = _best(pred, box, CONTAINER_TYPES)
        src = v if kind == "component" else (ctx.get("component") or {})
        want, want_v = src.get("name"), {str(k).lower(): str(x).lower() for k, x in (src.get("variant") or {}).items()}
        if kind == "ok" and not ctx.get("component"):
            return n is not None and iou >= thr
        if n is None or iou < thr:
            return want is None
        if n.component is None or want is None:
            return n.component is None and want is None
        got_v = {str(k).lower(): str(x).lower() for k, x in n.component.variant.items()}
        return n.component.name.lower() == want.lower() and all(got_v.get(k) == x for k, x in want_v.items())
    if kind == "ok":
        n, iou = _best(pred, box, (ctx.get("type"),) if ctx.get("type") else None)
        return n is not None and iou >= thr
    if kind == "text":
        n, iou = _best(pred, box, ("text",))
        return n is not None and iou >= thr and normalize_text(n.text) == normalize_text(v.get("text"))
    if kind == "icon":
        n, iou = _best(pred, box, ("icon",))
        return n is not None and iou >= thr and (n.icon_name or "") == v.get("icon_name")
    if kind == "extra":
        n, iou = _best(pred, box, (ctx.get("type"),) if ctx.get("type") else None)
        return n is None or iou < thr
    if kind == "missing":
        t = v.get("type")
        n, iou = _best(pred, box, (t,) if t in ("text", "icon", "image") else None)
        return n is not None and iou >= thr
    if kind == "geometry":
        n, iou = _best(pred, v["box"], (ctx.get("type"),) if ctx.get("type") else None)
        return n is not None and iou >= float(P["feedback.eval.geom_iou"])
    if kind == "color":
        from dt.ir import Color
        n, iou = _best(pred, box, (ctx.get("type"),) if ctx.get("type") else None)
        if n is None or iou < thr:
            return False
        c = n.text_style.color if n.type == "text" and n.text_style else n.fill_color
        if v.get("color") is None:
            return c is None
        return c is not None and c.delta_e(Color.from_hex(v["color"])) <= float(P["map.color_de"])
    if kind == "should_be_image":
        n, iou = _best(pred, box, ("image",))
        return n is not None and iou >= thr
    if kind == "should_be_editable":
        n, iou = _best(pred, box, ("image",))
        return not (n is not None and iou >= thr and n.image_ref)
    return None


def evaluate(home: Optional[str] = None, overrides: Optional[dict] = None, dirs: Optional[list[str]] = None,
             log: Logger = None) -> dict:
    """Current model (plus ``overrides``) on every user-corpus case: perceive -> map, structural metrics against
    the corrected gt (bench.run_case) and per-item agreement. Perception is cached per screenshot, so comparing
    two rule sets perceives each screen once."""
    from dt.mapping import map_document
    from dt.selftest import bench
    from dt.selftest.metrics import case_composite
    say = _say(log)
    dirs = corpus_dirs(home) if dirs is None else dirs
    rows, by_kind = [], {}
    for d, cid in bench.scan_cases(dirs)[0]:
        side = os.path.join(d, cid + ".feedback.json")  # promoted scenario seeds carry their items alongside
        try:
            if os.path.exists(side):
                with open(side) as f:
                    bundle = json.load(f)
            else:
                bundle = load_bundle(cid, home)
        except (OSError, ValueError):
            bundle = {"items": []}
        png = os.path.join(d, cid + ".png")
        perceived = _perceived(png)
        row = bench.run_case(d, cid, stages=("perceive", "map"), prediction=perceived, overrides=overrides)
        with bench.param_overrides(overrides):
            pred = map_document(Document.from_dict(perceived.to_dict()), bench._design_system())
        res = []
        for it in bundle.get("items") or []:
            ok = check_item(it, pred)
            if ok is None:
                continue
            res.append(ok)
            k = by_kind.setdefault(it["kind"], {"n": 0, "ok": 0})
            k["n"] += 1
            k["ok"] += int(ok)
        # component / token labels of nodes the user did not correct are the model's own unverified guesses: the
        # explicit corrections are scored by item accuracy, the rest of the gt guards geometry, text and paint
        unverified = dict(row["metrics"], component_acc=None, token_acc=None)
        rows.append({"id": cid, "composite": row.get("composite"), "component_acc": row["metrics"].get("component_acc"),
                     "composite_unlabelled": case_composite(unverified) if row["metrics"] else None,
                     "node_recall": row["metrics"].get("node_recall"), "items": len(res), "item_ok": sum(res),
                     "item_acc": (sum(res) / len(res)) if res else None, "error": row.get("error")})
        say(f"  [feedback eval] {cid}: composite={row.get('composite')} items {sum(res)}/{len(res)}")

    def mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    n_items = sum(r["items"] for r in rows)
    return {"n_cases": len(rows), "composite": mean("composite"), "composite_unlabelled": mean("composite_unlabelled"),
            "component_acc": mean("component_acc"),
            "node_recall": mean("node_recall"), "items": n_items,
            "item_acc": round(sum(r["item_ok"] for r in rows) / n_items, 4) if n_items else None,
            "by_kind": by_kind, "cases": rows}


def eval_history(home: Optional[str] = None, log: Logger = None) -> dict:
    """``dt feedback eval``: evaluate the current model on all feedback cases and append to history.jsonl."""
    build_corpus(home)
    res = evaluate(home, log=log)
    hp = history_path(home)
    prev = None
    if os.path.exists(hp):
        with open(hp) as f:
            lines = [ln for ln in f if ln.strip()]
        prev = json.loads(lines[-1]) if lines else None
    entry = {"ts": now_iso(), "model": model_version(home), **{k: v for k, v in res.items() if k != "cases"},
             "cases": [{k: r[k] for k in ("id", "composite", "item_acc", "items")} for r in res["cases"]]}
    os.makedirs(os.path.dirname(hp), exist_ok=True)
    with open(hp, "a") as f:
        f.write(json.dumps(entry) + "\n")
    entry["previous"] = {k: prev.get(k) for k in ("ts", "model", "composite", "item_acc", "n_cases")} if prev else None
    return entry


# --------------------------------------------------------------------------- the global gate
def bench_gate(overrides: dict, workers: int = 3, limit: Optional[int] = None, log: Logger = None) -> dict:
    """The shared regression gate (knowledge/baseline.json) with ``overrides`` applied in every bench worker."""
    from dt.selftest import bench
    say = _say(log)
    say("[feedback] global bench gate ...")
    rep = bench.run(list(bench.DEFAULT_CORPORA), limit=limit, workers=workers, out_root=None, overrides=overrides)
    base = bench.load_baseline(bench.BASELINE_PATH)
    fails = ["no knowledge/baseline.json"] if base is None else bench.check_regression(rep, base)
    return {"pass": not fails, "failures": fails, "composite": rep["composite"],
            "baseline_composite": (base or {}).get("composite"), "n_cases": rep["n_cases"]}


def _user_gate(before: dict, after: dict) -> Optional[bool]:
    if not before.get("n_cases"):
        return None
    eps = float(P["feedback.gate.user_eps"])
    ok = True
    for k in ("item_acc", "composite_unlabelled"):
        b, a = before.get(k), after.get(k)
        if b is not None and a is not None and a < b - eps:
            ok = False
    return ok


def _metrics(m: dict) -> dict:
    return {k: m.get(k) for k in ("n_cases", "items", "item_acc", "composite", "composite_unlabelled", "component_acc",
                                  "node_recall")}


# --------------------------------------------------------------------------- learn
def learn(home: Optional[str] = None, gate: bool = True, workers: int = 3, bench_limit: Optional[int] = None,
          log: Logger = None, bench_fn: Optional[Callable[..., dict]] = None) -> dict:
    """``dt feedback learn``: bundles -> user corpus + candidate local rules -> before/after on the user corpus ->
    global bench gate (unless ``gate=False``) -> accepted rules replace ``$DT_HOME/learned_rules.json``.
    Every rule change is recorded in ``$DT_HOME/ledger.jsonl`` whether accepted or not."""
    say = _say(log)
    home = home or dt_home()
    bundles = list_bundles(home)
    dirs = build_corpus(home, bundles)
    say(f"[feedback] {len(bundles)} bundles, {len(dirs)} user-corpus cases")
    rules = derive_rules(observations(bundles), scope="local")
    rejected = _read_rejected(home)
    for r in rules:  # a rule the gates rejected stays off until its evidence changes
        rej = rejected.get(r["id"])
        if r["active"] and rej is not None and rej.get("evidence") == _evidence_key(r):
            r["active"], r["rejected_by"] = False, rej.get("ledger")
    path = local_rules_path(home)
    old = read_rules(path)
    diff = rules_diff(old, rules)
    res: dict[str, Any] = {"bundles": len(bundles), "corpus": dirs, "rules": len(rules),
                           "active": sum(1 for r in rules if r["active"]), "diff": diff, "ledger": None}
    if _diff_empty(diff):
        res.update(changed=False, accepted=None)
        say("[feedback] no rule changes")
        return res

    def behaviour(rs: list[dict]) -> list:
        return sorted(json.dumps([r["id"], r.get("component"), r.get("variant") or {}], sort_keys=True) for r in rs if r.get("active"))
    if behaviour(old) == behaviour(rules):  # only evidence counts of inactive rules moved: bookkeeping, no gate
        write_rules(path, rules, "local")
        res.update(changed=True, accepted=True, behaviour_changed=False)
        say("[feedback] evidence updated; no active rule changed")
        return res
    res["behaviour_changed"] = True
    cand = path + ".candidate.json"
    write_rules(cand, rules, "local")
    try:
        before = evaluate(home, overrides={"map.learned_rules": "global," + path}, dirs=dirs, log=log)
        after = evaluate(home, overrides={"map.learned_rules": "global," + cand}, dirs=dirs, log=log)
        gates: dict[str, bool] = {}
        skipped: list[str] = []
        ug = _user_gate(before, after)
        if ug is None:
            skipped.append("user_corpus (no screenshots stored)")
        else:
            gates["user_corpus"] = ug
        bench_ev = None
        if gate:
            bench_ev = (bench_fn or bench_gate)({"map.learned_rules": "global," + cand}, workers=workers, limit=bench_limit, log=log)
            gates["bench"] = bool(bench_ev["pass"])
        else:
            skipped.append("bench (--no-gate)")
        accepted = all(gates.values())
        sources = sorted({s for r in diff["added"] for s in next((x["sources"] for x in rules if x["id"] == r["id"]), [])}
                         | {s for rid in diff["changed"] for s in next((x["sources"] for x in rules if x["id"] == rid), [])}
                         | {s for r in diff["removed"] for s in next((x.get("sources", []) for x in old if x["id"] == r["id"]), [])})
        extra = {"skipped": skipped, "model": model_version(home)}
        if bench_ev is not None:
            extra["bench"] = bench_ev
        entry = ledger.make_entry("rule", "local", "feedback", sources,
                                  {"file": "learned_rules.json", "rules": diff}, _metrics(before), _metrics(after), gates,
                                  accepted, extra_evidence=extra)
        trial = {x["id"] for x in diff["added"]} | set(diff["changed"])
        trial = {r["id"]: r for r in rules if r["id"] in trial and r["active"]}
        for rid, r in trial.items():
            if accepted:
                rejected.pop(rid, None)
            else:
                rejected[rid] = {"evidence": _evidence_key(r), "ledger": entry["id"]}
        _write_rejected(home, rejected)
        if accepted:
            snap = os.path.join(feedback_root(home), "rules_history", entry["id"] + ".json")
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            with open(snap, "w") as f:
                json.dump({"rules_before": old, "existed": os.path.exists(path)}, f, indent=2)
            os.replace(cand, path)
        ledger.append(entry, ledger.local_path(home))
        res.update(changed=True, accepted=accepted, gates=gates, before=_metrics(before), after=_metrics(after),
                   ledger=entry["id"], skipped=skipped, bench=bench_ev)
        say(f"[feedback] rules {'ACCEPTED' if accepted else 'rejected'}: gates {gates} (ledger {entry['id']})")
        return res
    finally:
        if os.path.exists(cand):
            os.remove(cand)


def _evidence_key(r: dict) -> list:
    return [r.get("component"), r.get("variant") or {}, int(r.get("support", 0)), int(r.get("against", 0))]


def _rejected_path(home: Optional[str]) -> str:
    return os.path.join(feedback_root(home), "rejected_rules.json")


def _read_rejected(home: Optional[str]) -> dict:
    try:
        with open(_rejected_path(home)) as f:
            return dict(json.load(f))
    except (OSError, ValueError):
        return {}


def _write_rejected(home: Optional[str], rejected: dict) -> None:
    os.makedirs(feedback_root(home), exist_ok=True)
    with open(_rejected_path(home), "w") as f:
        json.dump(rejected, f, indent=2)


def revert(entry_id: str, home: Optional[str] = None) -> dict:
    """Restore the local rules an accepted ledger entry replaced; appends a ``revert`` entry."""
    home = home or dt_home()
    entries = {e["id"]: e for e in ledger.read(ledger.local_path(home))}
    e = entries.get(entry_id)
    if e is None or e.get("kind") != "rule" or not e.get("accepted"):
        raise ValueError(f"{entry_id}: not an accepted local rule change in {ledger.local_path(home)}")
    snap = os.path.join(feedback_root(home), "rules_history", entry_id + ".json")
    with open(snap) as f:
        s = json.load(f)
    path = local_rules_path(home)
    cur = read_rules(path)
    if s.get("existed"):
        write_rules(path, s["rules_before"], "local")
    elif os.path.exists(path):
        os.remove(path)
    restored = s.get("rules_before") or []
    out = ledger.make_entry("revert", "local", "manual", [entry_id], {"file": "learned_rules.json", "rules": rules_diff(cur, restored)},
                            {"rules": len(cur), "active": sum(1 for r in cur if r.get("active"))},
                            {"rules": len(restored), "active": sum(1 for r in restored if r.get("active"))}, {}, True,
                            reverts=entry_id)
    ledger.append(out, ledger.local_path(home))
    return out


__all__ = ["build_corpus", "corpus_dirs", "observations", "derive_rules", "rules_diff", "read_rules", "write_rules",
           "evaluate", "eval_history", "check_item", "learn", "revert", "bench_gate", "bundle_target_path"]
