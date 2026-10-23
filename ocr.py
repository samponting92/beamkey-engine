"""
OCR fallback for sheets whose text was exported as drawn lines (no text layer).

Returns text lines in the same shape the engine uses for real PDF text:
    {"text": str, "bbox": (x0, y0, x1, y1) in PDF points, "dir": (dx, dy)}
Two passes: upright text, and text rotated to read bottom-to-top (vertical labels).
"""
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pymupdf as fitz

try:
    import pytesseract
    from PIL import Image
    AVAILABLE = True
except Exception:  # pragma: no cover
    AVAILABLE = False

DPI = 300
USEFUL = re.compile(r"^[A-Z]{1,3}\s*-?\s*\d{1,3}\b|\d\s*x\s*\d|UC|UB|SHS|RHS|PFC|C24|BEAM|TRIM|LINTEL|FLITCH", re.I)


def _lines_from(img, min_conf):
    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    groups = {}
    for i, w in enumerate(d["text"]):
        w = w.strip()
        try:
            conf = float(d["conf"][i])
        except ValueError:
            conf = -1
        if not w or conf < min_conf:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append((d["left"][i], d["top"][i], d["width"][i], d["height"][i], w))
    out = []
    for words in groups.values():
        words.sort()
        # split a tesseract "line" where the gap between words is large (separate labels)
        run = [words[0]]
        for w in words[1:] + [None]:
            if w is not None and w[0] - (run[-1][0] + run[-1][2]) < 2.2 * max(run[-1][3], w[3]):
                run.append(w); continue
            x0 = min(r[0] for r in run); y0 = min(r[1] for r in run)
            x1 = max(r[0] + r[2] for r in run); y1 = max(r[1] + r[3] for r in run)
            out.append((" ".join(r[4] for r in run), x0, y0, x1, y1))
            run = [w] if w is not None else []
    return out


CACHE = os.environ.get("BEAMKEY_OCR_CACHE", "/tmp/beamkey-ocr-cache")


def ocr_lines(page, dpi=DPI):
    if not AVAILABLE:
        return []
    key = hashlib.sha1(page.read_contents() + str((page.rect, dpi, 3)).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return [dict(l, bbox=tuple(l["bbox"]), dir=tuple(l["dir"])) for l in json.load(f)]
    out = _ocr(page, dpi)
    try:
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
    except OSError:
        pass
    return out


def _ocr(page, dpi):
    """Read the sheet in overlapping tiles (dense line work confuses whole-page OCR)."""
    s = dpi / 72
    W, H = page.rect.width, page.rect.height
    tw, th, ov = 420, 300, 40
    tiles = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            tiles.append(fitz.Rect(x, y, min(x + tw, W), min(y + th, H)))
            x += tw - ov
        y += th - ov

    def run(clip):
        pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        h = img.height
        res = []
        for t, x0, y0, x1, y1 in _lines_from(img, 55):
            res.append({"text": t, "bbox": (clip.x0 + x0 / s, clip.y0 + y0 / s, clip.x0 + x1 / s, clip.y0 + y1 / s),
                        "dir": (1.0, 0.0)})
        rot = img.rotate(-90, expand=True)  # text reading bottom-to-top becomes upright
        for t, x0, y0, x1, y1 in _lines_from(rot, 45):
            if USEFUL.search(t):  # rotated passes find noise; keep member-like text only
                # rotated (x', y') -> tile (y', h - x')
                res.append({"text": t, "bbox": (clip.x0 + y0 / s, clip.y0 + (h - x1) / s,
                                                 clip.x0 + y1 / s, clip.y0 + (h - x0) / s), "dir": (0.0, -1.0)})
        return res

    def whole():
        # whole-sheet pass at a lower resolution: different segmentation, catches what tiles miss
        ws = 250 / 72
        pix = page.get_pixmap(dpi=250, colorspace=fitz.csGRAY)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        h = img.height
        res = [{"text": t, "bbox": (x0 / ws, y0 / ws, x1 / ws, y1 / ws), "dir": (1.0, 0.0)}
               for t, x0, y0, x1, y1 in _lines_from(img, 55)]
        for t, x0, y0, x1, y1 in _lines_from(img.rotate(-90, expand=True), 45):
            if USEFUL.search(t):
                res.append({"text": t, "bbox": (y0 / ws, (h - x1) / ws, y1 / ws, (h - x0) / ws), "dir": (0.0, -1.0)})
        return res

    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        w_f = ex.submit(whole)
        found = [l for part in ex.map(run, tiles) for l in part]
        found += w_f.result()
    # drop duplicates from tile overlaps (same text, same place)
    out = []
    good = re.compile(r"^[A-Z]{1,3}\d{1,3}\b")
    for l in sorted(found, key=lambda l: (-len(l["text"]), not good.match(l["text"]))):
        cx = (l["bbox"][0] + l["bbox"][2]) / 2; cy = (l["bbox"][1] + l["bbox"][3]) / 2
        dup = False
        for o in out:
            ox = (o["bbox"][0] + o["bbox"][2]) / 2; oy = (o["bbox"][1] + o["bbox"][3]) / 2
            if o["dir"] == l["dir"] and abs(ox - cx) < 6 and abs(oy - cy) < 4 and (
                    l["text"] in o["text"] or len(l["text"]) == len(o["text"])):
                dup = True
                break
        if not dup:
            out.append(l)
    return out
