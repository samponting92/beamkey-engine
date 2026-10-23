"""
Optional Claude step: a short plain-English report per drawing set.

Placement never depends on this. If ANTHROPIC_API_KEY is not set, a basic
report is built from the engine output alone.
"""
import json
import os

import pymupdf as fitz

MODEL = os.environ.get("BEAMKEY_MODEL", "claude-sonnet-5-5")

PROMPT = """You are checking a structural engineer's drawing set for a contractor or QS.
Below is (1) the text on each sheet and (2) the members BeamKey coloured, unplaced labels
and suggestions. Write a short report in plain UK English, no em dashes. Return JSON only:
{"summary": [one line per sheet saying what is coloured],
 "judgement_calls": [provisional "allow"/"if required" items, existing items left uncoloured],
 "check_by_hand": [unplaced labels and unlabelled suggestions worth a look],
 "conflicts": [contradictions between notes and labels, e.g. steel grade, weld sizes,
               sheet status or dates, odd section sizes]}
Do not resolve conflicts, only report them. Keep each item to one line."""


def _basic(pages):
    summary, prov, check = [], [], []
    for i, p in enumerate(pages):
        cats = {}
        for m in p.get("members", []):
            cats[m["cat"]] = cats.get(m["cat"], 0) + 1
            if m.get("provisional"):
                prov.append(f"Sheet {i + 1}: {m['ref']} coloured as provisional")
        bits = [f"{n} {c}" for c, n in cats.items()]
        if p.get("posts"):
            bits.append(f"{len(p['posts'])} posts")
        summary.append(f"Sheet {i + 1}: " + (", ".join(bits) or "nothing coloured"))
        check += [f"Sheet {i + 1}: label not placed: {t}" for t in p.get("unplaced", [])[:8]]
        if p.get("suggestions"):
            check.append(f"Sheet {i + 1}: {len(p['suggestions'])} unlabelled centrelines to check")
    return {"summary": summary, "judgement_calls": prov, "check_by_hand": check,
            "conflicts": [], "source": "engine"}


def write_report(pdf_bytes, pages):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _basic(pages)
    try:
        import anthropic
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(f"=== SHEET {i + 1}\n{pg.get_text()[:6000]}" for i, pg in enumerate(doc))
        slim = [{"sheet": i + 1,
                 "members": [{"cat": m["cat"], "ref": m["ref"]} for m in p.get("members", [])],
                 "posts": len(p.get("posts", [])), "unplaced": p.get("unplaced", []),
                 "suggestions": len(p.get("suggestions", []))} for i, p in enumerate(pages)]
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL, max_tokens=1500,
            messages=[{"role": "user", "content": f"{PROMPT}\n\nSHEET TEXT:\n{text}\n\nBEAMKEY OUTPUT:\n{json.dumps(slim)}"}])
        raw = msg.content[0].text.strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        out = json.loads(raw)
        out["source"] = "claude"
        return out
    except Exception as e:
        out = _basic(pages)
        out["note"] = f"AI report unavailable ({type(e).__name__}); basic report shown."
        return out
