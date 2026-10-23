# BeamKey engine

The service behind beamkey.co.uk. It reads vector structural drawings and returns them colour coded:
red steel beams, orange steel trimmers, blue SHS/RHS and posts, green timber.

The geometry finds where each member is. The labels say what it is. No AI is used for placement,
so results are exact, fast (about 1.5 seconds for a 5-sheet set) and cheap.

## How it works

Reads drawings the way an engineer does: each note is read as a whole, and where a note has a
leader (a thin line ending in an arrowhead or dot) the member is wherever the arrow points.
Notes without a leader are matched to the member they are written along or sit beside.
The key is placed in genuinely empty space found by scanning the rendered page.

1. Reads every line in the PDF with its weight.
2. Finds chain/dashed centrelines (steel) and bundles of parallel lines (timber, flitches).
3. Reads the member labels (`152x152 UC30`, `TRIM F-02`, `2/150x50 C24 + 6mm plate`, `80x80 SHS`).
4. Learns which line weight carries the steel on each sheet, then matches each label to its line.
5. Infers unlabelled members that close an SHS frame, finds posts sitting on steel.
6. Returns anything it could not place, plus unlabelled centrelines, for a quick human check.

Sheets with no text layer (text exported as lines) are read with Tesseract OCR, in overlapping
tiles plus a whole-sheet pass, about 25 seconds a sheet. Beam and column schedules or legends
("B12 | 2.9 | 152x152x23UC", "C1-C4 - Frame Columns UC 152x152x30") are read so short codes resolve
to sizes, and OCR slips ("Bi" for "B1") are corrected only when the result is a scheduled code.
Grid lines (edge to edge, ending in a bubble) are ignored.

Results so far (23 Oct 2026):

| Set | Engineer style | Result |
|---|---|---|
| 69a Regent's Park Road | chain lines, full labels | 60 of 66 members exact, 0 false, 20 of 23 posts |
| 11 Birch Hollow | dashes drawn as filled shapes, B-codes + schedule, grid lines | first floor 47 of 52 B-beams, roof 22 of 25, FC columns as posts |
| Coach House | no text layer (OCR), legend codes | B1 x2, B2/B3, B4, roof frames RF1-7, ring beams; F1/F2 and angled C1-C4 labels missed |
| 245 Elgin Ave | architect's Building Regs set | correctly finds almost nothing: no steel sizes on it |
| Dunraven House, 41 Albert Bridge Rd | dash pairs, multi-line callouts, some leader lines, slab edges as dashed outlines | ground floor beams 1, 1A, 5, 6, 7A and PFC lintels; first floor ridge beams, roof frames, beam 9, 14B; beams 2, 3, 4, 7 left for review; 0 wrong colours seen |

The 69a benchmark must stay at 0 false members after any change. Run it yourself:

    python tests/benchmark.py "tests/sample/69a Regents Park Road structural revC.pdf" tests/sample/69a-checked-members.json

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | `{"ok": true}` |
| POST | `/analyse` | multipart `file` (PDF). Returns `job_id`, and per sheet: `width`, `height`, `image`, `members`, `posts`, `suggestions`, `unplaced`. Plus `warnings`. |
| GET | `/jobs/{id}/page/{n}.png?dpi=110` | Sheet image for the review screen |
| POST | `/jobs/{id}/render` | JSON `{"pages": [...]}` (edited members/posts per sheet). Returns the colour-coded PDF. Send `{}` to use the automatic result. |
| POST | `/jobs/{id}/report` | Short report. Uses Claude if `ANTHROPIC_API_KEY` is set, otherwise a basic one. |
| POST | `/process` | One shot: PDF in, colour-coded PDF out |

Coordinates are PDF points (1/72 inch), origin top-left, the same as `width` x `height`.
To draw them over the sheet image, use an SVG with `viewBox="0 0 {width} {height}"`.

Member: `{"cat": "beam|trimmer|shs|timber", "ref": "...", "pts": [[x,y],[x,y]], "provisional": false}`
Post: `{"ref": "...", "pt": [x, y]}`

## Settings (environment variables)

| Name | Default | Purpose |
|---|---|---|
| `BEAMKEY_API_KEY` | (none) | If set, callers must send header `X-BeamKey-Key` |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list, e.g. `https://beamkey.co.uk,https://www.beamkey.co.uk` |
| `ANTHROPIC_API_KEY` | (none) | Turns on the Claude-written report |
| `BEAMKEY_MODEL` | `claude-sonnet-5-5` | Model for the report |
| `BEAMKEY_MAX_MB` | `60` | Upload limit |
| `BEAMKEY_JOB_TTL_HOURS` | `24` | Uploaded drawings are deleted after this |

## Run locally

    pip install -r requirements.txt
    uvicorn app:app --reload

Open http://localhost:8000/docs to try every endpoint in the browser.

## Deploy

Railway or Render: connect this GitHub repo. Both detect the `Dockerfile` and run it.
Set the environment variables above, then add the custom domain `api.beamkey.co.uk`.

## Limits

- Scanned (image-only) PDFs are flagged; members on those sheets are added by hand in the review screen.
- Labels joined to a member only by a long leader line are not placed automatically.
  They appear in `unplaced`, and their line usually appears in `suggestions`.
- New engineers' drawing styles: add the set to `tests/sample` with a checked members file
  and run the benchmark before and after any change.
