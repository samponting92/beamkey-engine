"""
BeamKey engine.

Reads a vector structural drawing PDF and finds:
  - steel members: chain/dashed centrelines matched to steel labels
  - timber members: bundles of parallel solid lines matched to timber labels
  - posts: small heavy rectangles sitting on matched steel
  - suggestions: unlabelled centrelines at the same weight as matched steel

Geometry finds WHERE a member is. Labels say WHAT it is.
No AI is needed for placement.
"""
import math
import re
from collections import defaultdict

import pymupdf as fitz

# ---------------------------------------------------------------- categories
COLOURS = {
    "beam":    ((0.86, 0.10, 0.10), "Steel beams (UC / UB)"),
    "trimmer": ((1.00, 0.55, 0.00), "Steel trimmers (UC / UB)"),
    "shs":     ((0.10, 0.35, 0.95), "Steel columns, posts, SHS/RHS"),
    "timber":  ((0.10, 0.65, 0.20), "Timber trimmers, flitches, lintels"),
}
BAND_WIDTH = 6
BAND_OPACITY = 0.5
POST_SIZE = 10

# ---------------------------------------------------------------- label rules
STEEL_SECTION = re.compile(r"\b(UC|UB|PFC|RSJ)\b|\d\s*(UC|UB|PFC)(\b|\s*\d)", re.I)
HOLLOW = re.compile(r"\b(SHS|RHS|CHS)\b|\d\s*(SHS|RHS|CHS)", re.I)
TIMBER = re.compile(r"\bC(16|24)\b|C24|C16|FLITCH|TIMBER|\+\s*\d+\s*MM\s*PLATE|GLULAM|LVL|^\s*\d\s*/|"
                    r"\b\d{2,3}\s*X\s*(38|44|47|50|63|75)\b(?!\s*X)", re.I)
TRIM = re.compile(r"TRIM", re.I)
MEMBER_WORD = re.compile(r"\b(BEAM|TRIM\w*|LINTEL|FLITCH)\b", re.I)
REF = re.compile(r"\b[A-Z]{1,3}\s?-\s?\d{1,2}\b|\b[A-Z]{1,3}\d{2}\b")
TWO_NO = re.compile(r"\b(2\s*(No\.?|NUMBER|NR)\b|2\s*/\s*\d+\s*x\s*\d+\s*(UC|UB))|(UC|UB|PFC)\s*x\s*2\b|DOUBLE BEAM", re.I)
PROVISIONAL = re.compile(r"IF REQUIRED|\bALLOW\b|PROVISIONAL", re.I)
EXISTING = re.compile(r"\bEXIST\w*|\bEXT\b", re.I)
EXCLUDE = re.compile(
    r"STUD|JOIST|RAFTER|@|NOGGIN|DECK|\bWALL\b|BELOW|FLOOR|ROOF|SLOPE|STRAP|M305|"
    r"CATNIC|BXD|BDH|STONE|\bPC\b|PADSTONE|SPREADER PLATE|PARTY|BOLTS|HELIBAR|"
    r"COLUMN|POST|NOTE|SHOWING|CONTRACTOR|SUPPORT TO|CONNECT EVERY|HANGER|DESIGNED|"
    r"^\s*\d+\s*\.|^\s*\.|^ALLOW FOR|^NEW \d|COMBINATION|^ON \d|^\(|SCHEDULE|\bSPEC\b|REFERENCE|"
    r"^OVER\b|PICK UP|^NEW LINTEL$|GRIDLINE|\bCDP\b|^ITEM|\bUNDER\b|BEAM AND BLOCK|^FD\d|\bSTUBS?\b", re.I)
SPEC = re.compile(r"\d+\s*x\s*\d+.*?(UC|UB|PFC|SHS|RHS|CHS)|\d{2,3}\s*(UC|UB|SHS|RHS)\s*\d|"
                  r"(UC|UB)\s*\d{3}\s*x|C24|C16|FLITCH|GLULAM|\bLVL\b|PSL", re.I)
LEGEND = re.compile(r"^\s*([A-Z]{1,3}\s*\d{1,3}(?:\s*(?:-|,|&|TO)\s*[A-Z]{0,3}\s*\d{1,3})*)\s*[-=:]\s+(.+)$", re.I)
CODE = re.compile(r"^\s*([A-Z]{1,3})\s*-?\s*(\d{1,3})\b")
WREF = re.compile(r"^\s*((?:[A-Z]+\s+){0,2}(?:BEAM|LINTEL|FRAME|TRIMMER|TRIM|COLUMN|POST|PURLIN|RAFTER|JOIST|BINDER))\s*(\d{1,3}[A-Z]?)\b", re.I)


def wref_key(t):
    m = WREF.match(t)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1)).upper() + m.group(2).upper()


SOFT = re.compile(r"\b(ROOF|FLOOR|WALL|SLOPE)\b", re.I)


def classify_text(t):
    """Return (kind, steel_cat) where kind is steel/timber/either/None."""
    if re.match(r"^\s*ALL\b|^\s*BEAM SIZE|SPAN \(M\)", t, re.I):
        return None, None  # general notes and schedule headings
    test = t
    if STEEL_SECTION.search(t) or HOLLOW.search(t) or WREF.match(t):
        test = SOFT.sub("", t)   # "ROOF FRAME 1 100x100x10 SHS" names a member; roof/floor/wall don't veto it
    if len(t) > 70 or EXCLUDE.search(test):
        return None, None
    if EXISTING.search(t) and not PROVISIONAL.search(t):
        return None, None
    if HOLLOW.search(t):
        return "steel", "shs"
    if STEEL_SECTION.search(t):
        return "steel", "trimmer" if TRIM.search(t) else "beam"
    if TIMBER.search(t):
        return "timber", None
    if MEMBER_WORD.search(t) or (REF.fullmatch(t.strip().replace(" ", "")) is not None):
        return "either", "trimmer" if TRIM.search(t) else "beam"
    if REF.search(t) and len(t) < 16:
        return "either", "trimmer" if TRIM.search(t) else "beam"
    return None, None


# ---------------------------------------------------------------- geometry
def _segments(page, min_width=0.0):
    """All straight segments with their stroke width and dash flag."""
    out = []
    fills = defaultdict(list)   # dashes drawn as filled shapes, often two triangles per dash
    draws = page.get_drawings()
    for d in draws:
        w = d.get("width") or 0
        if d.get("type") == "f":
            r = d["rect"]
            if 1.0 <= max(r.width, r.height) <= 60 and min(r.width, r.height) <= 30:
                fills[(round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))].append(d)
            continue
        if w < min_width:
            continue
        dashed = bool(d.get("dashes")) and d.get("dashes") not in ("[] 0", "[] 0.0")
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                L = math.hypot(b.x - a.x, b.y - a.y)
                if L >= 0.4:
                    out.append((round(w, 2), a.x, a.y, b.x, b.y, L, dashed))
            elif it[0] == "re" and w >= 0.3:
                r = it[1]
                for (ax, ay, bx, by) in ((r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                                         (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)):
                    L = math.hypot(bx - ax, by - ay)
                    if L >= 0.4:
                        out.append((round(w, 2), ax, ay, bx, by, L, dashed))
    for group in fills.values():
        seg = _thin_fill(group)
        if seg:
            out.append(seg)
    return out


_TEXT_CACHE = {}


def text_lines(page):
    """Text lines with position and direction; OCR when the sheet has no text layer."""
    key = (id(page.parent), page.number)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = re.sub(r"\s+", " ", " ".join(sp["text"] for sp in ln["spans"])).strip()
            if t:
                out.append({"text": t, "bbox": tuple(ln["bbox"]), "dir": tuple(ln["dir"]), "block": tuple(b["bbox"])})
    if len(out) < 15 and len(page.get_cdrawings()) > 1500:
        from ocr import ocr_lines
        out = ocr_lines(page)
        _TEXT_CACHE[("ocr",) + key] = True
    _TEXT_CACHE[key] = out
    return out


def used_ocr(page):
    return _TEXT_CACHE.get(("ocr", id(page.parent), page.number), False)


def _thin_fill(group):
    """Filled shapes sharing one box (a dash drawn as two triangles) -> one segment."""
    pts = []
    for it in [it for d in group for it in d["items"]]:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "re":
            q = it[1]
            pts += [fitz.Point(q.x0, q.y0), fitz.Point(q.x1, q.y1), fitz.Point(q.x0, q.y1), fitz.Point(q.x1, q.y0)]
        elif it[0] == "qu":
            pts += list(it[1])
        else:
            return None
    if len(pts) < 3:
        return None
    cx = sum(p.x for p in pts) / len(pts); cy = sum(p.y for p in pts) / len(pts)
    sxx = sum((p.x - cx) ** 2 for p in pts); syy = sum((p.y - cy) ** 2 for p in pts)
    sxy = sum((p.x - cx) * (p.y - cy) for p in pts)
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(th), math.sin(th)
    along = [(p.x - cx) * ux + (p.y - cy) * uy for p in pts]
    across = [-(p.x - cx) * uy + (p.y - cy) * ux for p in pts]
    L = max(along) - min(along); T = max(across) - min(across)
    if T > 2.5 or L < 1.0 or L < 2 * T:
        return None
    # a dash fills its whole strip; an arrowhead (one triangle) fills about half of it
    area = 0.0
    for d in group:
        poly = []
        for it in d["items"]:
            if it[0] == "l":
                poly += [it[1], it[2]] if not poly else [it[2]]
            elif it[0] == "re":
                area += abs(it[1].width * it[1].height)
            elif it[0] == "qu":
                area += abs(it[1].rect.width * it[1].rect.height)
        if len(poly) >= 3:
            area += abs(sum(poly[i].x * poly[i - 1].y - poly[i - 1].x * poly[i].y for i in range(len(poly)))) / 2
    if T > 0.3 and area < 0.7 * L * T:
        return None
    a0, a1 = min(along), max(along)
    return (round(T + 0.01, 1), cx + a0 * ux, cy + a0 * uy, cx + a1 * ux, cy + a1 * uy, L, False)


def schedule_map(doc):
    """Read beam / column schedules: {'B12': {'spec': '152x152x23UC', 'column': False}, ...}."""
    out = {"_cells": {}}
    for page in doc:
        cells = out["_cells"].setdefault(page.number, [])
        lines = [(l["text"], fitz.Rect(l["bbox"])) for l in text_lines(page) if abs(l["dir"][0]) > 0.9]
        heads = [(t, r) for t, r in lines if "SCHEDULE" in t.upper()]
        for t, r in lines:
            # legend split into cells: "C1-C4" | "- Frame Columns" with "UC 152x152x30" underneath
            if not (len(t) <= 14 and re.fullmatch(r"[A-Z]{1,3}\s*\d{1,3}(\s*(-|,|&|TO)\s*[A-Z]{0,3}\s*\d{1,3})*", t.strip(), re.I)):
                continue
            yc = (r.y0 + r.y1) / 2
            right = sorted([(u, q) for u, q in lines if 0 <= q.x0 - r.x1 < 40 and abs((q.y0 + q.y1) / 2 - yc) < 4],
                           key=lambda m: m[1].x0)
            if not right or SPEC.search(right[0][0]) and not right[0][0].lstrip().startswith(("-", "=", ":")):
                continue
            desc, q = right[0]
            if not desc.lstrip().startswith(("-", "=", ":")):
                continue
            joined = desc
            if not SPEC.search(joined):
                under = [(u, w) for u, w in lines if 0 < w.y0 - q.y1 < 10 and abs(w.x0 - q.x0) < 20 and SPEC.search(u)]
                if under:
                    u, w = min(under, key=lambda m: m[1].y0)
                    joined += " " + u
                    cells.append(w)
            if SPEC.search(joined):
                cells += [r, q]
                for ref in _expand_refs(t):
                    out.setdefault(ref, {"spec": joined.lstrip("-=: ").strip(),
                                         "column": bool(re.search(r"COLUMN|POST", joined, re.I))})
        for i, (t, r) in enumerate(lines):
            # legend style on one line: "B1 - 203 x 203 x 52UC", "C1-C4 - Frame Columns ..."
            lg = LEGEND.match(t)
            if lg and _expand_refs(lg.group(1)):
                spec = lg.group(2).strip()
                if not SPEC.search(spec):
                    below = [(u, q) for u, q in lines if 0 < q.y0 - r.y1 < 10 and q.x0 > r.x0 and q.x0 < r.x1
                             and SPEC.search(u)]
                    if below:
                        spec = spec + " " + min(below, key=lambda b: b[1].y0)[0]
                        cells.append(min(below, key=lambda b: b[1].y0)[1])
                if SPEC.search(spec):
                    cells.append(r)
                    for ref in _expand_refs(lg.group(1)):
                        out.setdefault(ref, {"spec": spec, "column": bool(re.search(r"COLUMN|POST", spec, re.I))})
                    continue
            if len(t) > 40 or not (SPEC.search(t) or re.fullmatch(r"\s*(TBC|OMITTED|TBA)\s*", t, re.I)):
                continue
            yc = (r.y0 + r.y1) / 2
            mates = [(u, q) for u, q in lines if q.x1 <= r.x0 + 1 and q.x0 > r.x0 - 320
                     and abs((q.y0 + q.y1) / 2 - yc) < 3]
            mates = [m for m in mates if _expand_refs(m[0]) and r.x0 - m[1].x1 < 150]
            if not mates:
                continue
            nearest = max(mates, key=lambda m: m[1].x0)
            refs = _expand_refs(nearest[0])
            mates = [nearest]
            above = [h for h in heads if h[1].y1 <= r.y0 + 1 and abs(h[1].x0 - min(m[1].x0 for m in mates)) < 120]
            head = max(above, key=lambda h: h[1].y1)[0].upper() if above else ""
            col = "COLUMN" in head or "POST" in head
            cells += [r, nearest[1]]
            for ref in refs:
                out.setdefault(ref, {"spec": t, "column": col})
    # a prefix used mostly for scheduled columns (FC, C) marks unscheduled ones as columns too
    by = defaultdict(list)
    for k, v in out.items():
        if k != "_cells":
            by[re.match(r"[A-Z]+", k).group()].append(v["column"])
    out["_colprefix"] = {p for p, v in by.items() if sum(v) >= max(2, 0.5 * len(v))}
    return out


def _expand_refs(t):
    t = t.upper()
    w = re.match(r"^\s*((?:[A-Z]+\s+){0,2}(?:BEAM|LINTEL|FRAME|TRIMMER|TRIM|COLUMN|POST|PURLIN|RAFTER|JOIST|BINDER))"
                 r"\s*(\d{1,3}[A-Z]?(?:\s*(?:,|&|AND)\s*\d{1,3}[A-Z]?)*)\b", t)
    if w:
        head = re.sub(r"\s+", "", w.group(1))
        return [head + n for n in re.findall(r"\d{1,3}[A-Z]?", w.group(2))]
    m = re.match(r"^\s*([A-Z]{1,3})\s*(\d{1,3})\s*(?:-|TO)\s*([A-Z]{1,3})?\s*(\d{1,3})\b", t)
    if m and int(m.group(4)) >= int(m.group(2)) and int(m.group(4)) - int(m.group(2)) < 40:
        return [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(4)) + 1)]
    m = re.match(r"^\s*([A-Z]{1,3})\s*(\d{1,3}(?:\s*[,&]\s*\d{1,3})*)\b", t)
    if m:
        return [f"{m.group(1)}{n}" for n in re.findall(r"\d{1,3}", m.group(2))]
    return []


def _frame(theta):
    ux, uy = math.cos(theta), math.sin(theta)
    return ux, uy, -uy, ux


def centrelines(segs):
    """Group collinear short segments into dashed / chain lines."""
    buckets = defaultdict(list)
    for s in segs:
        w, x0, y0, x1, y1, L, dashed = s
        if L > 40 and not dashed:
            continue  # long solid lines are not chain lines
        th = math.atan2(y1 - y0, x1 - x0) % math.pi
        buckets[(w, round(math.degrees(th)) % 180)].append((th, s))
    lines = []
    for (w, deg), items in buckets.items():
        # merge neighbouring angle buckets
        for dd in (-1, 1):
            items = items + [i for i in buckets.get((w, (deg + dd) % 180), [])
                             if abs(math.degrees(i[0]) - deg) < 1.5]
        sx = sum(math.cos(2 * i[0]) for i in items); sy = sum(math.sin(2 * i[0]) for i in items)
        th = (math.atan2(sy, sx) / 2) % math.pi
        ux, uy, nx, ny = _frame(th)
        rows = []
        for _, s in items:
            _, x0, y0, x1, y1, L, _ = s
            o = x0 * nx + y0 * ny
            p0, p1 = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
            rows.append((o, p0, p1, L))
        rows.sort()
        # cluster by offset
        clusters, cur = [], [rows[0]] if rows else []
        for r in rows[1:]:
            if r[0] - cur[-1][0] <= 1.2:
                cur.append(r)
            else:
                clusters.append(cur); cur = [r]
        if cur:
            clusters.append(cur)
        for c in clusters:
            c.sort(key=lambda r: r[1])
            run = [c[0]]
            for r in c[1:] + [None]:
                if r is not None and r[1] - max(x[2] for x in run) <= 14:
                    run.append(r); continue
                if len(run) >= 3:
                    p0 = min(x[1] for x in run); p1 = max(x[2] for x in run)
                    span = p1 - p0
                    cover = sum(x[2] - x[1] for x in run)
                    if span > 25 and cover / span < 0.93:
                        o = sum(x[0] for x in run) / len(run)
                        lines.append(dict(w=w, th=th, off=o, p0=p0, p1=p1, n=len(run)))
                run = [r] if r is not None else []
    # de-duplicate lines found twice through neighbouring buckets
    uniq = []
    for l in sorted(lines, key=lambda l: -(l["p1"] - l["p0"])):
        if not any(abs(l["off"] - u["off"]) < 1.2 and abs(math.sin(l["th"] - u["th"])) < 0.03
                   and l["p0"] >= u["p0"] - 3 and l["p1"] <= u["p1"] + 3 for u in uniq):
            uniq.append(l)
    return uniq


def _endpoints(th, off, p0, p1):
    ux, uy, nx, ny = _frame(th)
    return [[round(p0 * ux + off * nx, 1), round(p0 * uy + off * ny, 1)],
            [round(p1 * ux + off * nx, 1), round(p1 * uy + off * ny, 1)]]


def _label_geom(lab, th):
    ux, uy, nx, ny = _frame(th)
    cx, cy = lab["c"]
    return cx * ux + cy * uy, cx * nx + cy * ny


def _parallel(a, b, tol_deg):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d) <= math.radians(tol_deg)


def match_steel(lab, lines, max_dist=22):
    cands = []
    for i, l in enumerate(lines):
        if not _parallel(lab["th"], l["th"], 10):
            continue
        along, off = _label_geom(lab, l["th"])
        if l["p0"] - 15 <= along <= l["p1"] + 15:
            dist = abs(off - l["off"])
            if dist <= max_dist:
                cands.append((dist, i))
    cands.sort()
    if not cands:
        return [], None
    if lab.get("pair"):
        best = [cands[0][1]]
        for d, i in cands[1:]:
            if abs(lines[i]["off"] - lines[best[0]]["off"]) <= 12:
                best.append(i); break
        return best, cands[0][0]
    return [cands[0][1]], cands[0][0]


def match_timber(lab, solid, max_dist=24):
    """Find a bundle of 2+ parallel solid lines near the label."""
    th = lab["th"]
    ux, uy, nx, ny = _frame(th)
    along, loff = _label_geom(lab, th)
    rows = []
    for s in solid:
        w, x0, y0, x1, y1, L, _ = s
        if L < 25:
            continue
        sth = math.atan2(y1 - y0, x1 - x0) % math.pi
        if not _parallel(sth, th, 6):
            continue
        o = ((x0 + x1) / 2) * nx + ((y0 + y1) / 2) * ny
        if abs(o - loff) > max_dist:
            continue
        p0, p1 = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
        if not (p0 - 15 <= along <= p1 + 15):
            continue
        rows.append((o, p0, p1))
    if len(rows) < 2:
        return None, None
    rows.sort()
    bundles, cur = [], [rows[0]]
    for r in rows[1:]:
        if 0.3 <= r[0] - cur[-1][0] <= 4.5 or abs(r[0] - cur[-1][0]) < 0.3:
            cur.append(r)
        else:
            bundles.append(cur); cur = [r]
    bundles.append(cur)
    best = None
    for b in bundles:
        offs = sorted(set(round(r[0], 1) for r in b))
        if len(offs) < 2:
            continue
        o = (offs[0] + offs[-1]) / 2
        maxlen = max(r[2] - r[1] for r in b)
        main = [r for r in b if r[2] - r[1] >= 0.6 * maxlen]
        p0 = min(r[1] for r in main); p1 = max(r[2] for r in main)
        dist = abs(o - loff)
        if best is None or dist < best[0]:
            best = (dist, o, p0, p1)
    if best is None:
        return None, None
    dist, o, p0, p1 = best
    # convert back to page coordinates via a pseudo-line
    return dict(th=th, off=o, p0=p0, p1=p1), dist


def _theta_from_dir(dx, dy):
    return math.atan2(dy, dx) % math.pi


_OCR_DIGIT = str.maketrans({"i": "1", "l": "1", "I": "1", "|": "1", "!": "1", "o": "0", "O": "0",
                            "s": "5", "S": "5", "z": "2", "Z": "2", "a": "4", "A": "4", "g": "9", "b": "6", "$": "5"})


def _ocr_fix(t, sched):
    """'Bi' -> 'B1', '( Ba' -> 'B4': accepted only if the result is a scheduled code."""
    t = t.strip(" ()[]{}.,:;'\"-")
    m = re.match(r"^([A-Z$]{1,3}?)([A-Za-z0-9|!$]{1,3})$", t)
    if not m:
        return None
    head = m.group(1).replace("$", "S")
    cand = head + m.group(2).translate(_OCR_DIGIT)
    mm = CODE.match(cand)
    if mm and f"{mm.group(1)}{int(mm.group(2))}" in sched:
        return cand
    return None


def _lookup(t, sched, colprefix):
    """Schedule entry for a note's reference: codes (B12, FC6) or words (BEAM 7B, TIMBER LINTEL 5)."""
    m = CODE.match(t)
    if m and not WREF.match(t):
        key = f"{m.group(1).upper()}{int(m.group(2))}"
        entry = sched.get(key)
        if m.group(1).upper() in colprefix and not (entry and entry["column"]):
            entry = {"spec": (entry or {}).get("spec", "column"), "column": True}
        return entry
    k = wref_key(t)
    return sched.get(k) if k else None


def labels(page, sched=None, columns=None):
    """One label per note. The first line or two name the member; the rest is description."""
    from notes import note_blocks, bbox_union
    out = []
    sched = sched or {}
    cells = sched.get("_cells", {}).get(page.number, [])
    colprefix = sched.get("_colprefix", set())
    lines = []
    for ln in text_lines(page):
        x0, y0, x1, y1 = ln["bbox"]
        if any(c.contains(fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)) for c in cells):
            continue
        t = ln["text"]
        if len(t) <= 8 and sched and not _lookup(t, sched, colprefix):
            fixed = _ocr_fix(t, sched)
            if fixed:
                ln = dict(ln, text=fixed)
        lines.append(ln)

    for blk in note_blocks(lines):
        first = blk[0]["text"]
        # header = first line, plus the second when the first names no size ("BEAM 17 - 2 NUMBER")
        hdr = blk[:1]
        if len(blk) > 1 and not (SPEC.search(first) or STEEL_SECTION.search(first) or HOLLOW.search(first)
                                 or re.search(r"\d{2,3}\s*X\s*\d{2,3}", first, re.I)):
            hdr = blk[:2]
        t = " ".join(l["text"] for l in hdr)
        x0, y0, x1, y1 = bbox_union(hdr)
        dx, dy = blk[0]["dir"]
        block_box = bbox_union(blk)

        two = " ".join(l["text"] for l in blk[:2])
        # column notes ("C5 UNDER / 150X100X10 RHS", "COLUMN OF BOX FRAME") are posts, not beams
        entry = _lookup(first, sched, colprefix)
        if (entry and entry["column"]) or re.match(r"^\s*(C\d+[A-Z]?\b|COLUMN|POST\b)", first, re.I) \
                or re.search(r"\bCOLUMN\b|\bPOST\b", two, re.I) and not re.search(r"\bBEAM\b|LINTEL|TRIM", first, re.I):
            if columns is not None and len(first) < 30:
                columns.append(dict(text=f"{first} ({entry['spec']})" if entry else first,
                                    c=((x0 + x1) / 2, (y0 + y1) / 2), bbox=fitz.Rect(x0, y0, x1, y1)))
            continue
        if entry and re.search(r"OMITTED", entry["spec"], re.I):
            continue
        sized = STEEL_SECTION.search(t) or HOLLOW.search(t)
        if EXISTING.search(first) and not (PROVISIONAL.search(t) and sized) or re.match(r"^\s*EX\b", first, re.I):
            continue  # existing structure stays uncoloured
        two = " ".join(l["text"] for l in blk[:2])
        if entry and not re.search(r"TBC", entry["spec"], re.I) and not EXCLUDE.search(first):
            kind, cat = classify_text(entry["spec"] + (" TRIM" if TRIM.search(t) else ""))
            t = f"{t} ({entry['spec']})" if entry["spec"] not in t else t
        else:
            kind, cat = classify_text(t)
            if not kind and wref_key(first):
                kind, cat = "either", "trimmer" if TRIM.search(first) else "beam"
            if kind == "either" and re.search(r"\bBEAM\b|PFC|\bUC\b|\bUB\b", t, re.I) and not TIMBER.search(t):
                kind = "steel"
        if not kind:
            continue
        if cat == "beam" and TRIM.search(two) and not re.search(r"TIMBER|C24|C16|\d\s*X\s*50\b", two, re.I):
            cat = "trimmer"
        out.append(dict(text=t, kind=kind, cat=cat, bbox=[x0, y0, x1, y1], block=block_box,
                        lines=[l["bbox"] for l in blk], dir=(dx, dy),
                        c=((x0 + x1) / 2, (y0 + y1) / 2), th=_theta_from_dir(dx, dy),
                        pair=bool(TWO_NO.search(t)), provisional=bool(PROVISIONAL.search(t))))
    return out


def posts_near(page, members):
    """Small heavy rectangles that sit on a matched steel member."""
    steel = [m for m in members if m["cat"] in ("beam", "trimmer", "shs")]
    found = []
    for d in page.get_drawings():
        if (d.get("width") or 0) < 0.6:
            continue
        r = d["rect"]
        big, small = max(r.width, r.height), min(r.width, r.height)
        if not (4 <= big <= 9.5 and small >= 3.5):
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        near = None
        for m in steel:
            (ax, ay), (bx, by) = m["pts"][0], m["pts"][-1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy or 1
            t = max(0, min(1, ((cx - ax) * vx + (cy - ay) * vy) / L2))
            dist = math.hypot(cx - (ax + t * vx), cy - (ay + t * vy))
            if dist <= 10:
                near = m; break
        if near and not any(math.hypot(cx - p["pt"][0], cy - p["pt"][1]) < 4 for p in found):
            found.append(dict(ref="post on " + near["ref"], pt=[round(cx, 1), round(cy, 1)]))
    return found


def _bundles_near(lab, segs, min_w, max_dist):
    """All bundles (2+ close parallel solid lines) near a label, nearest first."""
    th = lab["th"]
    ux, uy, nx, ny = _frame(th)
    along, loff = _label_geom(lab, th)
    rows = []
    for s in segs:
        w, x0, y0, x1, y1, L, dashed = s
        if dashed or w < min_w or L < 25:
            continue
        if not _parallel(math.atan2(y1 - y0, x1 - x0) % math.pi, th, 6):
            continue
        o = ((x0 + x1) / 2) * nx + ((y0 + y1) / 2) * ny
        if abs(o - loff) > max_dist:
            continue
        p0, p1 = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
        if p0 - 15 <= along <= p1 + 15:
            rows.append((o, p0, p1))
    rows.sort()
    out, cur = [], rows[:1]
    for r in rows[1:] + [None]:
        if r is not None and r[0] - cur[-1][0] <= 4.5:
            cur.append(r); continue
        offs = sorted(set(round(x[0], 1) for x in cur))
        if len(offs) >= 2 and offs[-1] - offs[0] >= 1.0:
            o = (offs[0] + offs[-1]) / 2
            maxlen = max(x[2] - x[1] for x in cur)
            main = [x for x in cur if x[2] - x[1] >= 0.6 * maxlen]
            out.append((abs(o - loff), dict(th=th, off=o, p0=min(x[1] for x in main),
                                              p1=max(x[2] for x in main))))
        cur = [r] if r is not None else []
    return sorted(out, key=lambda x: x[0])


def _coincident(a, b, tol=3.0):
    if not _parallel(a["th"], b["th"], 4):
        return False
    ux, uy, nx, ny = _frame(a["th"])
    # express b in a's frame using b's midpoint
    (x0, y0), (x1, y1) = _endpoints(b["th"], b["off"], b["p0"], b["p1"])
    ob = ((x0 + x1) / 2) * nx + ((y0 + y1) / 2) * ny
    if abs(ob - a["off"]) > tol:
        return False
    q0, q1 = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
    ov = min(a["p1"], q1) - max(a["p0"], q0)
    return ov > 0.5 * min(a["p1"] - a["p0"], q1 - q0)


def _dedupe(members):
    """Merge members drawn twice along the same line (keep the longer, join the refs)."""
    def info(m):
        (ax, ay), (bx, by) = m["pts"][0], m["pts"][-1]
        th = math.atan2(by - ay, bx - ax) % math.pi
        ux, uy, nx, ny = _frame(th)
        p = sorted((ax * ux + ay * uy, bx * ux + by * uy))
        return dict(th=th, off=((ax + bx) / 2) * nx + ((ay + by) / 2) * ny, p0=p[0], p1=p[1])
    rank = {"shs": 3, "trimmer": 2, "beam": 1, "timber": 0}
    out = []
    for m in sorted(members, key=lambda m: -math.dist(m["pts"][0], m["pts"][-1])):
        for o in out:
            a, b = info(o), info(m)
            if _parallel(a["th"], b["th"], 4) and _coincident(a, b, 3.5):
                if m["ref"] not in o["ref"] and "inferred" not in m["ref"]:
                    o["ref"] = o["ref"] + " / " + m["ref"]
                if m["cat"] != o["cat"] and "timber" not in (m["cat"], o["cat"]):
                    o["cat"] = max(m["cat"], o["cat"], key=rank.get)
                o["provisional"] = o["provisional"] or m["provisional"]
                break
        else:
            out.append(m)
    return out


def _ltype(lab):
    if lab["kind"] == "steel":
        return "shs" if lab["cat"] == "shs" else "uc"
    return lab["kind"]  # timber / either


def column_posts(page, columns):
    """Scheduled column labels (FC6, C3...) -> the small marker drawn next to each."""
    marks = []
    for d in page.get_drawings():
        r = d["rect"]
        big, small = max(r.width, r.height), min(r.width, r.height)
        if 2.5 <= big <= 11 and small >= 0.6 * big:
            marks.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    out = []
    for c in columns:
        b = c["bbox"]
        best = None
        for (x, y) in marks:
            dx = max(b.x0 - x, 0, x - b.x1); dy = max(b.y0 - y, 0, y - b.y1)
            dd = math.hypot(dx, dy)
            if dd <= 14 and (best is None or dd < best[0]):
                best = (dd, x, y)
        if best and not any(math.hypot(best[1] - p["pt"][0], best[2] - p["pt"][1]) < 3 for p in out):
            out.append(dict(ref=c["text"], pt=[round(best[1], 1), round(best[2], 1)]))
    return out


def _grid_bubbles(page):
    """Centres of grid bubbles: small circles holding a 1-3 character grid reference."""
    words = [l["bbox"] + (l["text"],) for l in text_lines(page)]
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if not (8 <= r.width <= 34 and abs(r.width - r.height) < 2.5):
            continue
        if not any(it[0] == "c" for it in d["items"]):
            continue
        inside = [w[4] for w in words if r.contains(fitz.Rect(w[:4]).tl + (fitz.Rect(w[:4]).br - fitz.Rect(w[:4]).tl) * 0.5)]
        if inside and all(len(w) <= 3 for w in inside):
            out.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width / 2))
    return out


def _is_gridline(l, page, bubbles):
    span = l["p1"] - l["p0"]
    ext = page.rect.width * abs(math.cos(l["th"])) + page.rect.height * abs(math.sin(l["th"]))
    if span > 0.5 * ext:
        return True
    for (x, y) in _endpoints(l["th"], l["off"], l["p0"], l["p1"]):
        for (bx, by, br) in bubbles:
            if math.hypot(x - bx, y - by) <= br + 12:
                return True
    return False


def analyse_page(page, sched=None):
    segs = _segments(page)
    bubbles = _grid_bubbles(page)
    cl = [l for l in centrelines(segs) if not _is_gridline(l, page, bubbles)]
    columns = []
    labs = labels(page, sched, columns)

    # pass 1: which line weight carries the steel on this sheet
    weights = defaultdict(int)
    close = set()
    for lab in labs:
        if lab["kind"] == "steel":
            idx, d = match_steel(lab, cl)
            for i in idx:
                weights[cl[i]["w"]] += 1
                if d is not None and d <= 12:
                    close.add(cl[i]["w"])
    if weights:
        top = max(weights.values())
        steel_w = {w for w, n in weights.items() if n >= max(2, 0.3 * top)} or {max(weights, key=weights.get)}
        steel_w |= {w for w in close if w >= max(steel_w)}  # heavier lines confirmed by a close steel label
        cls = [l for l in cl if l["w"] in steel_w]
    else:
        cls = cl
    geoms = [dict(kind="line", labels=[], **{k: l[k] for k in ("th", "off", "p0", "p1")}) for l in cls]

    def geom_for(b, kind="bundle"):
        for g in geoms:
            if _coincident(g, b):
                if kind == "box" and (g["p1"] - g["p0"]) > 1.3 * (b["p1"] - b["p0"]):
                    continue  # a box keeps its own length; don't merge it into a longer line
                if kind == "line" and (b["p1"] - b["p0"]) > 1.3 * (g["p1"] - g["p0"]) and _parallel(g["th"], b["th"], 2):
                    g["p0"], g["p1"] = min(g["p0"], b["p0"]), max(g["p1"], b["p1"])  # full length of the member
                return g
        g = dict(kind=kind, labels=[], **b)
        geoms.append(g)
        return g

    from notes import thin_segments, find_leaders, bundles_at, point_seg_dist, boxes, box_at, box_beside
    thin, heads = thin_segments(page)
    rects = boxes(page)
    led = set()
    leaders = {id(lab): find_leaders(lab["block"], thin, heads, lab.get("lines"),
                                     tuple(lab.get("dir", (1, 0))), segs)
               for lab in labs if lab.get("block")}
    lead_paths = [path for ls in leaders.values() for (_, _, _, path) in ls]

    def on_leader(s):
        a, b = (s[1], s[2]), (s[3], s[4])
        for path in lead_paths:
            for p_, q_ in zip(path, path[1:]):
                if point_seg_dist(a, p_, q_) < 1.5 and point_seg_dist(b, p_, q_) < 1.5:
                    return True
        return False
    segs_nl = [s_ for s_ in segs if not on_leader(s_)]

    def leader_cands(lab, t):
        """Follow the note's leader(s); return members found at the arrow tip."""
        found = []
        for tip, start, length, path in leaders.get(id(lab), []):
            here = []
            bx = box_at(tip, rects)
            if bx:
                here.append((0.5, geom_for(bx, "box")))   # a lintel or section drawn as a box right at the tip
            for l in cl:
                a, b = _endpoints(l["th"], l["off"], l["p0"], l["p1"])
                d = point_seg_dist(tip, a, b)
                if d <= 10 and l["p1"] - l["p0"] > 15:
                    here.append((d, geom_for({k: l[k] for k in ("th", "off", "p0", "p1")}, "line")))
            if not here or t in ("timber", "either"):
                for bnd in bundles_at(tip, segs_nl):
                    a, b = _endpoints(bnd["th"], bnd["off"], bnd["p0"], bnd["p1"])
                    d = point_seg_dist(tip, a, b)
                    pen = 0 if t in ("timber", "either") else 3
                    here.append((d + pen, geom_for(bnd)))
            if not here:
                # a member drawn as one solid line (common for steel on some practices' plans)
                for s_ in segs_nl:
                    w_, x0_, y0_, x1_, y1_, L_, dashed_ = s_
                    if L_ < 40 or w_ < 0.5:
                        continue
                    d = point_seg_dist(tip, (x0_, y0_), (x1_, y1_))
                    if d <= 8:
                        th_ = math.atan2(y1_ - y0_, x1_ - x0_) % math.pi
                        ux_, uy_, nx_, ny_ = _frame(th_)
                        p0_, p1_ = sorted((x0_ * ux_ + y0_ * uy_, x1_ * ux_ + y1_ * uy_))
                        here.append((d + 5, geom_for(dict(th=th_, off=x0_ * nx_ + y0_ * ny_, p0=p0_, p1=p1_), "line")))
            if not here:
                # a short member drawn with only one or two heavy dashes (e.g. a 1.2 m double PFC)
                heavy = [s_ for s_ in segs_nl if s_[0] >= 1.2 and s_[5] >= 6]
                hit = [s_ for s_ in heavy if point_seg_dist(tip, (s_[1], s_[2]), (s_[3], s_[4])) <= 6]
                if hit:
                    s0 = min(hit, key=lambda s_: point_seg_dist(tip, (s_[1], s_[2]), (s_[3], s_[4])))
                    th_ = math.atan2(s0[4] - s0[2], s0[3] - s0[1]) % math.pi
                    ux_, uy_, nx_, ny_ = _frame(th_)
                    o_ = s0[1] * nx_ + s0[2] * ny_
                    run = sorted([s_ for s_ in heavy if _parallel(math.atan2(s_[4] - s_[2], s_[3] - s_[1]) % math.pi, th_, 4)
                                  and abs(((s_[1] + s_[3]) / 2) * nx_ + ((s_[2] + s_[4]) / 2) * ny_ - o_) < 1.5],
                                 key=lambda s_: min(s_[1] * ux_ + s_[2] * uy_, s_[3] * ux_ + s_[4] * uy_))
                    p0_, p1_ = sorted((s0[1] * ux_ + s0[2] * uy_, s0[3] * ux_ + s0[4] * uy_))
                    for s_ in run:   # extend through collinear dashes with small gaps
                        q0, q1 = sorted((s_[1] * ux_ + s_[2] * uy_, s_[3] * ux_ + s_[4] * uy_))
                        if q0 <= p1_ + 10 and q1 >= p0_ - 10:
                            p0_, p1_ = min(p0_, q0), max(p1_, q1)
                    here.append((1, geom_for(dict(th=th_, off=o_, p0=p0_, p1=p1_), "line")))
            found += here
        found.sort(key=lambda c: c[0])
        return found

    # candidates for every label
    plan = []
    for lab in labs:
        t = _ltype(lab)
        cands = []
        # a size written right along a steel line is that member; otherwise follow the leader
        along_steel = False
        for l in cls:
            if _parallel(lab["th"], l["th"], 10):
                along, off = _label_geom(lab, l["th"])
                ux_, uy_, _, _ = _frame(l["th"])
                bx0, by0, bx1, by1 = lab["bbox"]
                ends = sorted((bx0 * ux_ + by0 * uy_, bx1 * ux_ + by1 * uy_))
                if l["p0"] - 8 <= ends[0] and ends[1] <= l["p1"] + 8 and abs(off - l["off"]) <= 16:
                    along_steel = True   # the whole label sits along this steel line
                    break
        lc = leader_cands(lab, t) if lab.get("block") and not along_steel else []
        if lc:
            # the leader says where the member is: trust it over anything drawn near the text
            led.add(id(lab))
            lab["leader"] = True
            plan.append((lc[0][0] - 100, lab, t, lc))
            continue
        if t in ("uc", "shs", "timber", "either"):
            for i, l in enumerate(cls):
                if not _parallel(lab["th"], l["th"], 10):
                    continue
                along, off = _label_geom(lab, l["th"])
                d = abs(off - l["off"])
                if lab.get("block") and d > 12:
                    # multi-line callout: measure from the edge of the whole text block
                    bx0, by0, bx1, by1 = lab["block"]
                    _, _, nx, ny = _frame(l["th"])
                    offs = [bx0 * nx + by0 * ny, bx1 * nx + by0 * ny, bx0 * nx + by1 * ny, bx1 * nx + by1 * ny]
                    if not (min(offs) <= l["off"] <= max(offs)):
                        d = min(d, min(abs(o - l["off"]) for o in offs) + 6)
                if l["p0"] - 15 <= along <= l["p1"] + 15 and d <= 22:
                    cands.append((d, geoms[i]))
        if t in ("timber", "either"):
            for d, b in _bundles_near(lab, segs, 0.3, 24)[:3]:
                cands.append((d, geom_for(b)))
        if t in ("uc", "shs") and not cands:
            for d, b in _bundles_near(lab, segs, 0.1, 16)[:2]:
                cands.append((d + 8, geom_for(b)))
        is_lintel = re.search(r"LINTEL", lab["text"], re.I)
        short_member = any(c[0] <= 24 and c[1]["p1"] - c[1]["p0"] <= 150 for c in cands)
        if lab.get("block") and MEMBER_WORD.search(lab["text"]) and (not cands or is_lintel and not short_member):
            bb = box_beside(lab["block"], rects)
            if bb:
                # a lintel is the box beside its note, not the floor lines running past it
                cands = [(c[0] + 8, c[1]) for c in cands]
                cands.append((bb[0], geom_for(bb[1], "box")))
        cands.sort(key=lambda c: c[0])
        plan.append((cands[0][0] if cands else 1e9, lab, t, cands))

    plan.sort(key=lambda p: p[0])
    unplaced = []
    for _, lab, t, cands in plan:
        taken = 0
        want = 2 if lab["pair"] else 1
        first = None
        for d, g in cands:
            types = {_ltype(x) for x in g["labels"]} - {"either"}
            if t != "either" and types and t not in types:
                continue
            if t == "shs" and "uc" in types or t == "uc" and "shs" in types:
                continue
            if taken and (abs(g["off"] - first["off"]) > 12 or g is first):
                continue
            g["labels"].append(lab)
            first = first or g
            taken += 1
            if taken == want:
                break
        if not taken and t != "either":
            unplaced.append(lab["text"])

    members = []
    for g in geoms:
        if not g["labels"]:
            continue
        types = {_ltype(x) for x in g["labels"]}
        if "timber" in types and not types & {"uc", "shs"}:
            cat = "timber"
        elif "shs" in types:
            cat = "shs"
        elif "uc" in types or g["kind"] == "line":
            cat = "trimmer" if any(TRIM.search(x["text"]) for x in g["labels"]) else "beam"
        else:
            cat = "timber"
        members.append(dict(cat=cat, ref=" / ".join(dict.fromkeys(x["text"] for x in g["labels"])),
                            pts=_endpoints(g["th"], g["off"], g["p0"], g["p1"]),
                            provisional=any(x["provisional"] for x in g["labels"]),
                            source=g["kind"]))

    # unlabelled centrelines that close an SHS frame (both ends on SHS members) are SHS
    shs = [m for m in members if m["cat"] == "shs"]
    def _on(pt, m, tol=7):
        (ax, ay), (bx, by) = m["pts"][0], m["pts"][-1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy or 1
        t = max(0, min(1, ((pt[0] - ax) * vx + (pt[1] - ay) * vy) / L2))
        return math.hypot(pt[0] - ax - t * vx, pt[1] - ay - t * vy) <= tol
    for g in geoms:
        if g["kind"] == "line" and not g["labels"] and len(shs) >= 1:
            a, b = _endpoints(g["th"], g["off"], g["p0"], g["p1"])
            if any(_on(a, m, 12) for m in shs) and any(_on(b, m, 12) for m in shs) \
                    and not any(_on(a, m, 3) and _on(b, m, 3) for m in members):
                g["labels"].append(dict(text="SHS frame (unlabelled)", kind="steel", cat="shs"))
                members.append(dict(cat="shs", ref="SHS frame member (unlabelled, inferred)",
                                    pts=[a, b], provisional=False, source="inferred"))

    members = _dedupe(members)
    posts = posts_near(page, members)
    for p in column_posts(page, columns):
        if not any(math.hypot(p["pt"][0] - q["pt"][0], p["pt"][1] - q["pt"][1]) < 5 for q in posts):
            posts.append(p)
    suggestions = [dict(cat="unknown", ref="Unlabelled centreline: check",
                        pts=_endpoints(g["th"], g["off"], g["p0"], g["p1"]))
                   for g in geoms if g["kind"] == "line" and not g["labels"]
                   and g["p1"] - g["p0"] > 30 and weights]
    steel_n = sum(m["cat"] in ("beam", "trimmer", "shs") for m in members)
    suggestions = sorted(suggestions, key=lambda g: -math.dist(*g["pts"]))[:25] if steel_n >= 2 else []
    return dict(width=page.rect.width, height=page.rect.height, members=members,
                posts=posts, suggestions=suggestions, unplaced=unplaced,
                vector=len(segs) > 50)


def analyse(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    _TEXT_CACHE.clear()
    for page in doc:
        if page.rotation:
            page.remove_rotation()
    sched = schedule_map(doc)
    pages = []
    for page in doc:
        res = analyse_page(page, sched)
        res["ocr"] = used_ocr(page)
        pages.append(res)
    _TEXT_CACHE.clear()
    return pages


# ---------------------------------------------------------------- overlay
def _key_spot(page, w, h, pad=8):
    """Find genuinely empty space for the key: render the page, then search for a blank
    window (key plus a margin) with no ink at all. Prefers corners, then edges, then anywhere."""
    import numpy as np
    dpi = 36
    sc = dpi / 72
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = (img < 235).astype(np.int32)
    ii = np.pad(ink.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    kw, kh = int((w + 2 * pad) * sc) + 1, int((h + 2 * pad) * sc) + 1
    H, W = ink.shape
    margin = int(20 * sc)
    if kw >= W - 2 * margin or kh >= H - 2 * margin:
        return fitz.Rect(30, 30, 30 + w, 30 + h)
    ys = np.arange(margin, H - margin - kh)
    xs = np.arange(margin, W - margin - kw)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    sums = ii[Y + kh, X + kw] - ii[Y, X + kw] - ii[Y + kh, X] + ii[Y, X]
    free = np.argwhere(sums == 0)
    if len(free):
        # closest to a page corner wins (keys are expected at corners; keeps the drawing area clear)
        cy = ys[free[:, 0]] + kh / 2; cx = xs[free[:, 1]] + kw / 2
        d = np.minimum.reduce([np.hypot(cx, cy), np.hypot(W - cx, cy), np.hypot(cx, H - cy), np.hypot(W - cx, H - cy)])
        k = int(np.argmin(d))
        y0, x0 = ys[free[k, 0]], xs[free[k, 1]]
    else:
        # nowhere fully clear: take the window covering the least ink
        k = np.unravel_index(int(np.argmin(sums)), sums.shape)
        y0, x0 = ys[k[0]], xs[k[1]]
    X0 = x0 / sc + pad; Y0 = y0 / sc + pad
    return fitz.Rect(X0, Y0, X0 + w, Y0 + h)


def _draw_key(page, rect, used):
    page.draw_rect(rect, color=(0.3, 0.3, 0.3), fill=(1, 1, 1), width=0.6)
    x, y = rect.x0 + 5, rect.y0 + 10
    page.insert_text((x, y), "BEAMKEY", fontsize=6.5, fontname="hebo")
    y += 9
    for cat in used:
        c, t = COLOURS[cat]
        page.draw_line((x + 1, y), (x + 21, y), color=c, width=6, stroke_opacity=BAND_OPACITY)
        page.insert_text((x + 25, y + 2.1), t, fontsize=5.6, fontname="helv")
        y += 10
    page.insert_text((x, y + 1), "Squares = columns / posts. Guidance overlay only;",
                     fontsize=4.4, fontname="helv", color=(0.3, 0.3, 0.3))
    page.insert_text((x, y + 7), "read with the original drawings.",
                     fontsize=4.4, fontname="helv", color=(0.3, 0.3, 0.3))


def render(pdf_bytes, pages):
    """pages: list of {members, posts} (as returned by analyse, after any user edits)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pn, page in enumerate(doc):
        if page.rotation:
            page.remove_rotation()
        data = pages[pn] if pn < len(pages) else {}
        used = []
        for m in data.get("members", []):
            if m.get("cat") not in COLOURS:
                continue
            sh = page.new_shape()
            sh.draw_polyline([tuple(p) for p in m["pts"]])
            sh.finish(color=COLOURS[m["cat"]][0], width=BAND_WIDTH, stroke_opacity=BAND_OPACITY,
                      lineCap=1, closePath=False)
            sh.commit()
            if m["cat"] not in used:
                used.append(m["cat"])
        for p in data.get("posts", []):
            x, y = p["pt"]
            s = POST_SIZE / 2
            page.draw_rect(fitz.Rect(x - s, y - s, x + s, y + s), color=COLOURS["shs"][0],
                           fill=COLOURS["shs"][0], fill_opacity=0.55, stroke_opacity=0.8, width=0.6)
        if data.get("posts") and "shs" not in used:
            used.append("shs")
        if not used:
            continue
        used.sort(key=list(COLOURS).index)
        h = 22 + 10 * len(used) + 8
        _draw_key(page, _key_spot(page, 152, h), used)
    return doc.tobytes(garbage=3, deflate=True)


def page_png(pdf_bytes, pn, dpi=110):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[pn]
    if page.rotation:
        page.remove_rotation()
    return page.get_pixmap(dpi=dpi).tobytes("png")
