"""
Notes and leaders: how an engineer actually reads a structural drawing.

A note is a block of text lines ("BEAM 7B - 254X254X73UC / ALLOW FOR WELDED HEAD
PLATE ..."). Its first line or two name the member; the rest is description.
If a leader line runs from the note, the member is wherever the leader points,
not whatever happens to be drawn next to the text.
"""
import math
import re

import pymupdf as fitz

STARTS_NEW = re.compile(r"^\s*[A-Z]{1,3}\s*-?\s*\d{1,3}[A-Z]?\b|^\s*BEAM\s*\d"
                        r"|^\s*(RIDGE|HIP|VALLEY|ROOF|CONSERV\w*|STEEL)\s+(BEAM|FRAME|LINTEL)\s*\d", re.I)


_DRAW_CACHE = {}


def drawings(page):
    """drawings(page), read once per page (it is the slow part of reading a sheet)."""
    key = (id(page.parent), page.number)
    if key not in _DRAW_CACHE:
        if len(_DRAW_CACHE) > 4:
            _DRAW_CACHE.clear()
        _DRAW_CACHE[key] = page.get_drawings()
    return _DRAW_CACHE[key]


# ---------------------------------------------------------------- note blocks
def _frame(dx, dy):
    """Reading direction u and the 'next line' direction n for a text line."""
    return (dx, dy), (-dy, dx)


def _proj(bbox, v):
    x0, y0, x1, y1 = bbox
    ps = [x0 * v[0] + y0 * v[1], x1 * v[0] + y0 * v[1], x0 * v[0] + y1 * v[1], x1 * v[0] + y1 * v[1]]
    return min(ps), max(ps)


def note_blocks(lines):
    """Group text lines into notes: same direction, left-aligned, stacked closely."""
    recs = []
    for i, ln in enumerate(lines):
        dx, dy = ln["dir"]
        dx, dy = round(dx), round(dy)
        if (dx, dy) not in ((1, 0), (0, -1), (0, 1), (-1, 0)):
            dx, dy = ln["dir"]
        u, n = _frame(dx, dy)
        s0, s1 = _proj(ln["bbox"], u)
        n0, n1 = _proj(ln["bbox"], n)
        recs.append(dict(i=i, u=u, s0=s0, s1=s1, n0=n0, n1=n1, h=max(n1 - n0, 1.0)))
    recs.sort(key=lambda r: (r["u"], r["n0"]))
    blocks = []
    open_blocks = []
    for r in recs:
        best = None
        new_member = bool(STARTS_NEW.match(lines[r["i"]]["text"]))
        for b in ([] if new_member else open_blocks):
            last = b[-1]
            if last["u"] != r["u"]:
                continue
            gap = r["n0"] - last["n1"]
            h = max(last["h"], r["h"])
            if not (0.6 <= r["h"] / last["h"] <= 1.6) or gap < -0.4 * h:
                continue
            left = abs(r["s0"] - last["s0"]) <= 6
            right = abs(r["s1"] - last["s1"]) <= 4
            centred = abs((r["s0"] + r["s1"]) / 2 - (last["s0"] + last["s1"]) / 2) <= 4
            if not ((left or right) and gap <= 0.95 * h or centred and gap <= 0.6 * h):
                continue
            if len(b) >= 2:
                # a clearly bigger gap than the note's own line spacing starts the next note
                inner_gap = max(b[k + 1]["n0"] - b[k]["n1"] for k in range(len(b) - 1))
                if gap > max(2.5 * inner_gap, 0.45 * h):
                    continue
            best = b
            break
        if best is not None:
            best.append(r)
        else:
            nb = [r]
            blocks.append(nb)
            open_blocks.append(nb)
        # blocks whose last line is far above can no longer grow
        open_blocks = [b for b in open_blocks if r["n0"] - b[-1]["n1"] < 40 or b[-1]["u"] != r["u"]]
    return [[lines[r["i"]] for r in b] for b in blocks]


def bbox_union(items):
    xs0 = [l["bbox"][0] for l in items]; ys0 = [l["bbox"][1] for l in items]
    xs1 = [l["bbox"][2] for l in items]; ys1 = [l["bbox"][3] for l in items]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# ---------------------------------------------------------------- leaders
def thin_segments(page):
    """Candidate leader segments: thin solid strokes in small drawings. Plus small filled arrowheads."""
    segs, heads = [], []
    for d in drawings(page):
        items = d["items"]
        if d.get("type") in ("f", "fs"):
            r = d["rect"]
            if 1.5 <= max(r.width, r.height) <= 10 and len(items) <= 6:
                if all(it[0] == "l" for it in items):          # arrowhead
                    pts = []
                    for it in items:
                        pts += [it[1], it[2]]
                    heads.append(pts)
                elif all(it[0] == "c" for it in items) and max(r.width, r.height) <= 5:   # dot
                    heads.append([fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)])
            if d.get("type") == "f":
                continue
        w = d.get("width") or 0
        dashes = d.get("dashes")
        if w > 1.0 or (dashes and dashes not in ("[] 0", "[] 0.0")) or len(items) > 8:
            continue
        col = tuple(round(c, 2) for c in (d.get("color") or ()))
        for it in items:
            if it[0] == "l":
                a, b = it[1], it[2]
                if math.dist(a, b) >= 2:
                    segs.append((a.x, a.y, b.x, b.y, round(w, 2), col))
    return segs, heads


def _near(p, q, tol):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def _head_at(pt, prev, heads):
    """Tip of an arrowhead or dot touching pt, or None."""
    for pts in heads:
        if any(_near((p.x, p.y), pt, 3) for p in pts):
            far = max(pts, key=lambda p: math.dist((p.x, p.y), prev))
            return (far.x, far.y)
    return None


def _has_partner(a, b, all_segs, pen=None):
    """True if a parallel line in the same pen runs alongside a-b a few points away
    (a member drawn as a double line, not a leader)."""
    L = math.dist(a, b)
    if L < 4:
        return False
    th = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
    ux, uy = math.cos(th), math.sin(th)
    nx, ny = -uy, ux
    o = a[0] * nx + a[1] * ny
    p0, p1 = sorted((a[0] * ux + a[1] * uy, b[0] * ux + b[1] * uy))
    for s in all_segs:
        x0, y0, x1, y1 = s[0], s[1], s[2], s[3]
        if pen is not None and (abs(s[4] - pen[0]) > 0.05 or s[5] != pen[1]):
            continue
        th2 = math.atan2(y1 - y0, x1 - x0) % math.pi
        d = abs(th2 - th)
        if min(d, math.pi - d) > math.radians(3):
            continue
        o2 = ((x0 + x1) / 2) * nx + ((y0 + y1) / 2) * ny
        if not (0.4 <= abs(o2 - o) <= 4.5):
            continue
        q0, q1 = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
        if min(p1, q1) - max(p0, q0) >= 0.8 * L:
            return True
    return False


_NODE_CACHE = {}


def _endpoint_index(segs):
    """Segment ends bucketed on a 2-point grid (with neighbours), so a leader can be followed
    from one segment to the next without scanning every line on the sheet."""
    key = id(segs)
    hit = _NODE_CACHE.get(key)
    if hit is not None and hit[0] is segs:
        return hit[1]
    idx = {}
    for j, sg in enumerate(segs):
        for (x, y) in ((sg[0], sg[1]), (sg[2], sg[3])):
            gx, gy = round(x / 2), round(y / 2)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    lst = idx.setdefault((gx + ddx, gy + ddy), [])
                    if not lst or lst[-1] != j:
                        lst.append(j)
    _NODE_CACHE.clear()
    _NODE_CACHE[key] = (segs, idx)
    return idx


def find_leaders(block, segs, heads, lines=None, direction=(1, 0), all_segs=None):
    """Leader paths leaving a note: off the end of one of its lines, or straight off its top or
    bottom. Lines running along the text or touching its side are underlines and box edges.
    Returns list of (tip, start, length, path)."""
    x0, y0, x1, y1 = block

    class _Box:   # plain-number point-in-box test (pymupdf's is slow when called a million times)
        def __init__(self, a, b, c, d):
            self.r = (a, b, c, d)

        def contains(self, p):
            return self.r[0] <= p[0] <= self.r[2] and self.r[1] <= p[1] <= self.r[3]
    inner = _Box(x0 - 7, y0 - 7, x1 + 7, y1 + 7)
    outer = _Box(x0 - 12, y0 - 12, x1 + 12, y1 + 12)
    nodes = _endpoint_index(segs)
    ux, uy = direction
    nx, ny = -uy, ux
    lines = lines or [block]
    spans = []
    for (lx0, ly0, lx1, ly1) in lines:
        su = sorted((lx0 * ux + ly0 * uy, lx1 * ux + ly1 * uy))
        sn = sorted((lx0 * nx + ly0 * ny, lx1 * nx + ly1 * ny))
        spans.append((su, sn))
    n_lo = min(sp[1][0] for sp in spans); n_hi = max(sp[1][1] for sp in spans)
    out = []
    for i, sg in enumerate(segs):
        ax, ay, bx, by = sg[:4]
        pen = sg[4:6]
        a, b = (ax, ay), (bx, by)
        for start, end in ((a, b), (b, a)):
            if not outer.contains(start):
                continue
            sx, sy = end[0] - start[0], end[1] - start[1]
            sl = math.hypot(sx, sy) or 1
            if not inner.contains(start) and abs(sx * ux + sy * uy) / sl <= 0.8:
                continue   # only a leader running on from a line end may start a little way off
            stub = inner.contains(end)
            if stub and (abs(sx * ux + sy * uy) / sl <= 0.8 or sl < 4):
                continue   # only a short run off a line end may stay near the note before turning away
            sp_u = start[0] * ux + start[1] * uy
            sp_n = start[0] * nx + start[1] * ny
            ep_u = end[0] * ux + end[1] * uy
            ep_n = end[0] * nx + end[1] * ny
            ok = False
            if abs(sx * ux + sy * uy) / sl > 0.8:
                # runs with the text: must leave from the end of a line, at that line's height
                for (su, sn) in spans:
                    if sn[0] - 4 <= sp_n <= sn[1] + 4:
                        # (some leaders start a little under the last word)
                        if (sp_u <= su[0] + 14 and ep_u < su[0] - 3 and ep_u < sp_u) or \
                                (sp_u >= su[1] - 14 and ep_u > su[1] + 3 and ep_u > sp_u):
                            ok = True
                            break
            else:
                # leaves across the text: must start above or below the note, not beside it
                ok = (sp_n <= n_lo + 2 and ep_n < sp_n) or (sp_n >= n_hi - 2 and ep_n > sp_n)
            if not ok:
                continue
            # walk the polyline from the far end, stopping at the first arrowhead or dot
            path = [start, end]
            used = {i}
            length = math.dist(start, end)
            cur = end
            tip = _head_at(cur, path[-2], heads)
            for _ in range(2):   # a leader is 1 to 3 straight segments
                if tip is not None:
                    break
                nxt = None
                for j in nodes.get((round(cur[0] / 2), round(cur[1] / 2)), ()):
                    cx, cy, dx_, dy_ = segs[j][:4]
                    if j in used:
                        continue
                    c, d = (cx, cy), (dx_, dy_)
                    if _near(c, cur, 1.2):
                        nxt = (j, d)
                    elif _near(d, cur, 1.2):
                        nxt = (j, c)
                    if nxt:
                        break
                if not nxt:
                    break
                used.add(nxt[0])
                length += math.dist(cur, nxt[1])
                cur = nxt[1]
                path.append(cur)
                tip = _head_at(cur, path[-2], heads)
            along_text = abs(sx * ux + sy * uy) / sl > 0.8
            # a very short arrow is only believed when it runs straight on from the end of a line
            if length < (5 if along_text else 10) or length > 260 or (tip is not None and fitz.Rect(block).contains(fitz.Point(tip))):
                continue
            if stub and (tip is None or math.dist(tip, start) < 25):
                continue
            if any(_has_partner(p_, q_, segs, pen) for p_, q_ in zip(path, path[1:])):
                continue  # part of a double-line member drawing, not a leader
            if tip is not None:
                out.append((tip, start, length, path))
    return out


# ---------------------------------------------------------------- what is at the tip
def point_seg_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy or 1e-9
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy)


def bundles_at(tip, segs_all, max_d=9):
    """Members drawn as 2+ close parallel solid lines passing near a point (timber, lintels, steel outlines)."""
    near = []
    for s in segs_all:
        w, x0, y0, x1, y1, L, dashed = s
        if w < 0.25 or L < 12:
            continue
        if point_seg_dist(tip, (x0, y0), (x1, y1)) <= max_d:
            near.append(s)
    groups = []
    for s in near:
        th = math.atan2(s[4] - s[2], s[3] - s[1]) % math.pi
        for g in groups:
            d = abs(g["th"] - th) % math.pi
            if min(d, math.pi - d) < math.radians(3):
                g["segs"].append(s)
                break
        else:
            groups.append({"th": th, "segs": [s]})
    out = []
    for g in groups:
        th = g["th"]
        ux, uy = math.cos(th), math.sin(th)
        nx, ny = -uy, ux
        rows = []
        for s in g["segs"]:
            o = ((s[1] + s[3]) / 2) * nx + ((s[2] + s[4]) / 2) * ny
            p0, p1 = sorted((s[1] * ux + s[2] * uy, s[3] * ux + s[4] * uy))
            rows.append((o, p0, p1))
        offs = sorted(set(round(r[0], 1) for r in rows))
        if len(offs) < 2 or offs[-1] - offs[0] < 1.0:
            continue
        # keep the tightest cluster around the tip
        tip_o = tip[0] * nx + tip[1] * ny
        rows.sort(key=lambda r: abs(r[0] - tip_o))
        clus = [rows[0]]
        for r in rows[1:]:
            if min(abs(r[0] - c[0]) for c in clus) <= 6:
                clus.append(r)
        o_lo, o_hi = min(c[0] for c in clus), max(c[0] for c in clus)
        if o_hi - o_lo < 1.0:
            continue
        maxlen = max(c[2] - c[1] for c in clus)
        main = [c for c in clus if c[2] - c[1] >= 0.5 * maxlen]
        p0, p1 = min(c[1] for c in main), max(c[2] for c in main)
        tip_p = tip[0] * ux + tip[1] * uy
        if not (p0 - 6 <= tip_p <= p1 + 6):
            continue  # the arrow must land on the member, not beside its end
        out.append(dict(th=th, off=(o_lo + o_hi) / 2, p0=p0, p1=p1))
    return out


def boxes(page):
    """Thin rectangles (lintels, steel sections drawn as boxes): drawn as rectangles, or as a
    closed outline of four straight lines."""
    out = []

    def keep(r):
        short, long_ = min(r.width, r.height), max(r.width, r.height)
        if 2.0 <= short <= 14 and long_ >= 2.5 * short and long_ >= 15:
            out.append(r)

    for d in drawings(page):
        items = d["items"]
        for it in items:
            if it[0] == "re":
                keep(fitz.Rect(it[1]))
            elif it[0] == "qu":
                keep(it[1].rect)
        # a rectangle drawn as 4 axis-aligned lines that close on themselves
        if 4 <= len(items) <= 5 and all(it[0] == "l" for it in items):
            pts = [it[1] for it in items] + [items[-1][2]]
            axis = all(abs(it[1].x - it[2].x) < 0.3 or abs(it[1].y - it[2].y) < 0.3 for it in items)
            if axis and math.dist(pts[0], pts[-1]) < 1.0:
                keep(fitz.Rect(min(p.x for p in pts), min(p.y for p in pts), max(p.x for p in pts), max(p.y for p in pts)))
    return out


def box_at(tip, rects, tol=8):
    """The thin rectangle the arrow tip touches, as a member along its long axis. Pairs of
    parallel boxes (a double PFC drawn one box per channel) become one member, and collinear
    boxes either side of a pier are joined, as an engineer would read them."""
    hit = [r for r in rects if fitz.Rect(r.x0 - tol, r.y0 - tol, r.x1 + tol, r.y1 + tol).contains(fitz.Point(tip))]
    if not hit:
        return None
    def dist(q):
        dx = max(q.x0 - tip[0], 0, tip[0] - q.x1); dy = max(q.y0 - tip[1], 0, tip[1] - q.y1)
        return math.hypot(dx, dy)
    r = min(hit, key=lambda q: (round(dist(q), 1), q.width * q.height))
    horiz = r.width >= r.height
    def axis(q):   # (offset of centre line, start, end, thickness)
        return ((q.y0 + q.y1) / 2, q.x0, q.x1, q.height) if horiz else ((q.x0 + q.x1) / 2, q.y0, q.y1, q.width)
    o, a0, a1, t = axis(r)
    same = [q for q in rects if (q.width >= q.height) == horiz]
    group = [q for q in same if abs(axis(q)[0] - o) <= 12 and abs(axis(q)[3] - t) <= 2]
    # a double member (two channels drawn as two boxes side by side) is read as one; a single box
    # (a lintel, a padstone) stays exactly as drawn
    partner = [q for q in group if 1.5 <= abs(axis(q)[0] - o) <= 12 and min(axis(q)[2], a1) - max(axis(q)[1], a0) > 0.5 * (a1 - a0)]
    if not partner:
        group = [r]
    # grow along the axis through collinear pieces with gaps up to 40 (a pier or wall)
    changed = True
    while changed:
        changed = False
        for q in group:
            qo, q0, q1, qt = axis(q)
            if q0 <= a1 + 40 and q1 >= a0 - 40 and (q0 < a0 or q1 > a1):
                a0, a1 = min(a0, q0), max(a1, q1)
                changed = True
    used = [q for q in group if axis(q)[1] >= a0 - 1 and axis(q)[2] <= a1 + 1]
    offs = [axis(q)[0] for q in used] or [o]
    off = (min(offs) + max(offs)) / 2
    if horiz:
        return dict(th=0.0, off=off, p0=a0, p1=a1)
    return dict(th=math.pi / 2, off=-off, p0=a0, p1=a1)


def box_beside(block, rects, gap=16):
    """A thin box right beside a note with no leader (a lintel labelled where it sits)."""
    x0, y0, x1, y1 = block
    zone = fitz.Rect(x0 - gap, y0 - gap, x1 + gap, y1 + gap)
    best = None
    for r in rects:
        short, long_ = min(r.width, r.height), max(r.width, r.height)
        if long_ < 4 * short or long_ < 25 or not zone.intersects(r) or fitz.Rect(block).intersects(r):
            continue
        dx = max(r.x0 - x1, x0 - r.x1, 0); dy = max(r.y0 - y1, y0 - r.y1, 0)
        d = math.hypot(dx, dy)
        if best is None or d < best[0]:
            best = (d, r)
    if not best:
        return None
    d, r = best
    if r.width >= r.height:
        return d, dict(th=0.0, off=(r.y0 + r.y1) / 2, p0=r.x0, p1=r.x1)
    return d, dict(th=math.pi / 2, off=-(r.x0 + r.x1) / 2, p0=r.y0, p1=r.y1)
