"""
Generate bbox_reveal_viz.html — a self-contained interactive visualization
showing placement bounding boxes from the annotation pipeline, revealed on mouse-over.

Green boxes  = pipeline proposals with positive ImageReward score (good placements)
Red boxes    = pipeline proposals with negative ImageReward score (bad placements)

The source data is opa_train_annotation_pipeline_low_iou_top1.jsonl:
each entry is a sliding-window bbox candidate that was:
  1. Proposed by our placement model
  2. Inserted into the background via Qwen-Image-Edit
  3. Scored by ImageReward

Usage:
    python generate_bbox_reveal_viz.py [--out bbox_reveal_viz.html]

Requires:
    - src/webapp_interactive_annotations/opa_train_annotation_pipeline_low_iou_top1.jsonl
    - Dataset_CVPR26/eval_our_pipeline_on_opa/train/combos.json
    - data/OPA/background/<class>/<bg_id>.jpg
"""

import argparse
import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config — edit these to change which backgrounds are shown
# ---------------------------------------------------------------------------
CHOSEN = [
    ("384507", "boat"),
    ("538604", "keyboard"),
    ("272940", "person"),
    ("205613", "motorcycle"),
    ("189868",  "cow"),
    ("190885", "car"),
]
# Set CHOSEN = None to show ALL combos (sorted by green count).
CONF_THRESHOLD = 0.40  # Grounding DINO confidence to count as detected (green)
REVEAL_TOP_N   = 5    # how many closest bboxes to reveal at once (per class)
IMG_SIZE       = 512  # center-crop target (px)
JPEG_QUALITY   = 90
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parents[2]
SCRIPT_DIR    = Path(__file__).parent
BBOXES_JSON   = ROOT / "data/HiddenObjects/bboxes.json"         # 1004 sliding-window grid
TOP1_JSONL    = SCRIPT_DIR / "opa_train_annotation_pipeline_low_iou_top2.jsonl"  # detected entries
COMBOS_JSON   = ROOT / "Dataset_CVPR26/eval_our_pipeline_on_opa/train/combos.json"
BG_DIR        = ROOT / "data/OPA/background"
IMG_CACHE_DIR = SCRIPT_DIR / "bbox_viz_data" / "images"         # pre-saved 512x512 crops


def center_crop(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Center-crop to square using min dimension, then resize to size×size."""
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top  = (h - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size)).resize(
        (size, size), Image.BILINEAR
    )


def load_detections() -> tuple[dict, dict]:
    """
    Load top1.jsonl (detected entries only) indexed by (combo_idx, bbox_id).
    The full 1004-grid is in bboxes.json; anything not in top1 is "not detected".
    Also returns the combo→(bg_id, fg_class) map.
    """
    combos = json.load(open(COMBOS_JSON))
    bg_combo_map = {(c["bg_id"], c["fg_class"]): c["combo_idx"] for c in combos}

    detections: dict[tuple, dict] = {}
    with open(TOP1_JSONL) as f:
        for line in f:
            d = json.loads(line)
            detections[(d["combo_idx"], d["bbox_id"])] = d
    print(f"  Loaded {len(detections)} entries from top1.jsonl")
    return detections, bg_combo_map


def encode_image(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def discover_all_combos(detections: dict, bg_combo_map: dict) -> list[tuple[str, str]]:
    """Find all combos that have detections and a valid bg image, sorted by green count."""
    import os
    all_bboxes = json.load(open(BBOXES_JSON))

    scored = []
    for (bg_id, fg_class), cidx in bg_combo_map.items():
        img_path = BG_DIR / fg_class / f"{bg_id}.jpg"
        if not img_path.exists():
            continue
        green = sum(
            1 for bb in all_bboxes
            if (det := detections.get((cidx, bb["bbox_id"])))
            and det.get("detected_bbox")
            and (det.get("confidence") or 0) > CONF_THRESHOLD
        )
        scored.append((bg_id, fg_class, green))

    scored.sort(key=lambda x: -x[2])
    print(f"  Discovered {len(scored)} combos with images")
    return [(bg_id, cls) for bg_id, cls, _ in scored]


def build_card_data(detections: dict, bg_combo_map: dict, chosen: list[tuple[str, str]]) -> list[dict]:
    # The full 1004-entry sliding window grid (same for all combos)
    all_bboxes = json.load(open(BBOXES_JSON))  # list of {bbox_id, x1, y1, x2, y2}

    cards = []
    for i, (bg_id, fg_class) in enumerate(chosen):
        cidx = bg_combo_map[(bg_id, fg_class)]

        good_bboxes = []  # detected_bbox where confidence > threshold
        bad_bboxes  = []  # input_bbox where no detection or confidence <= threshold

        for bb in all_bboxes:
            bid = bb["bbox_id"]
            det = detections.get((cidx, bid))
            if det and det.get("detected_bbox") and (det.get("confidence") or 0) > CONF_THRESHOLD:
                # Green: show the refined detected bbox
                d = det["detected_bbox"]
                good_bboxes.append([d["x1"], d["y1"], d["x2"], d["y2"]])
            else:
                # Red: show the input (proposed) bbox
                bad_bboxes.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])

        # Always use/save images from IMG_CACHE_DIR
        cached = IMG_CACHE_DIR / f"{bg_id}_{fg_class}.jpg"
        if not cached.exists():
            IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            img = center_crop(Image.open(BG_DIR / fg_class / f"{bg_id}.jpg"))
            img.save(cached, format="JPEG", quality=JPEG_QUALITY)
        img_b64 = encode_image(Image.open(cached))

        cards.append({
            "bg_id":       bg_id,
            "fg_class":    fg_class,
            "combo_idx":   cidx,
            "img_b64":     img_b64,
            "good_bboxes": good_bboxes,
            "bad_bboxes":  bad_bboxes,
        })
        if (i + 1) % 50 == 0 or i == len(chosen) - 1:
            print(f"  [{i+1}/{len(chosen)}] {bg_id} {fg_class}: green={len(good_bboxes)} red={len(bad_bboxes)}")

    return cards


def build_html(cards: list[dict]) -> str:
    cards_html_parts = []
    js_data = []

    for i, item in enumerate(cards):
        uid = f"card{i}"
        js_data.append({
            "uid":      uid,
            "fg_class": item["fg_class"],
            "good":     item["good_bboxes"],
            "bad":      item["bad_bboxes"],
        })
        cards_html_parts.append(f"""      <div class="viz-card">
        <div class="canvas-wrap" id="{uid}-wrap">
          <img class="bg-img" src="data:image/jpeg;base64,{item['img_b64']}" alt="{item['fg_class']} background" draggable="false">
          <canvas class="overlay-canvas" id="{uid}-canvas"></canvas>
        </div>
        <div class="viz-label">Place a <strong>{item['fg_class']}</strong> here?</div>
        <div class="viz-meta">combo&nbsp;{item['combo_idx']} &nbsp;·&nbsp; bg&nbsp;{item['bg_id']}</div>
        <div class="viz-legend">
          <span class="dot good-dot"></span> Detected &nbsp;
          <span class="dot bad-dot"></span> Not detected
        </div>
      </div>""")

    cards_js = json.dumps(js_data, separators=(",", ":"))
    cards_html_str = "\n".join(cards_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object Placement Suitability</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro&display=swap">
<style>
  body {{ font-family: 'Noto Sans', sans-serif; background: #fff; color: #363636; }}
  .hero {{ background: #fff; }}
  .section-title {{
    font-family: 'Google Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: #363636;
  }}
  .section-subtitle {{
    font-size: 0.95rem;
    color: #7a7a7a;
    margin-bottom: 2rem;
  }}
  .viz-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    max-width: 1080px;
    margin: 0 auto;
  }}
  .viz-card {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }}
  .viz-meta {{ font-size: 0.72rem; color: #aaa; text-align: center; font-family: monospace; }}
  .viz-label {{
    font-size: 0.9rem;
    color: #4a4a4a;
    text-align: center;
  }}
  .canvas-wrap {{
    position: relative;
    cursor: crosshair;
    border-radius: 6px;
    overflow: hidden;
    width: 100%;
    line-height: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  }}
  .bg-img {{ display: block; width: 100%; height: auto; user-select: none; }}
  .overlay-canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
  .viz-legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #7a7a7a;
  }}
  .dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; }}
  .good-dot {{ background: #23d160; }}
  .bad-dot  {{ background: #ff3860; }}
</style>
</head>
<body>
<section class="section">
  <div class="container is-max-desktop">
    <div class="has-text-centered" style="margin-bottom: 2rem;">
      <h2 class="section-title">Object Placement Suitability</h2>
      <p class="section-subtitle">
        Move your mouse over a scene to reveal placement candidates from our pipeline —
        <span style="color:#23d160; font-weight:600;">green</span> = object detected (conf&nbsp;&gt;&nbsp;0.40) &nbsp;·&nbsp;
        <span style="color:#ff3860; font-weight:600;">red</span> = not detected
      </p>
    </div>
    <div class="viz-grid">
{cards_html_str}
    </div>
  </div>
</section>
<script>
(function() {{
  const CARDS    = {cards_js};
  const IMG_SIZE = 512;    // native image pixels (bboxes are in this space)
  const TOP_N    = {REVEAL_TOP_N};  // reveal this many closest bboxes per class

  CARDS.forEach(function(c) {{
    const wrap   = document.getElementById(c.uid + '-wrap');
    const canvas = document.getElementById(c.uid + '-canvas');
    const ctx    = canvas.getContext('2d');
    let mouseX = null, mouseY = null, rafId = null;

    function resize() {{ canvas.width = wrap.clientWidth; canvas.height = wrap.clientHeight; draw(); }}
    function scale()    {{ return canvas.width / IMG_SIZE; }}
    function toNative(v) {{ return v / scale(); }}

    function bboxDist(bb, nx, ny) {{
      const cx = (bb[0]+bb[2])/2, cy = (bb[1]+bb[3])/2;
      return Math.sqrt((nx-cx)**2 + (ny-cy)**2);
    }}

    function topN(list, nx, ny) {{
      return list
        .map(bb => ({{ bb, d: bboxDist(bb, nx, ny) }}))
        .sort((a, b) => a.d - b.d)
        .slice(0, TOP_N);
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (mouseX === null) return;
      const nx = toNative(mouseX), ny = toNative(mouseY);
      const s  = scale();

      // Normalise distances within the revealed set so closest = full opacity
      function drawBboxes(list, color, fill) {{
        const items = topN(list, nx, ny);
        if (!items.length) return;
        const maxD = Math.max(...items.map(x => x.d)) || 1;
        items.forEach(function({{ bb, d }}) {{
          const t     = 1 - d / (maxD * 1.5 + 1); // softer falloff
          const alpha = 0.9 * (t*t*(3-2*t));
          const cx = (bb[0]+bb[2])/2, cy = (bb[1]+bb[3])/2;
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.fillStyle = fill;
          ctx.beginPath();
          ctx.roundRect(bb[0]*s, bb[1]*s, (bb[2]-bb[0])*s, (bb[3]-bb[1])*s, 3);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = color;
          ctx.beginPath(); ctx.arc(cx*s, cy*s, 3.5, 0, Math.PI*2); ctx.fill();
          ctx.restore();
        }});
      }}

      drawBboxes(c.bad,  'rgb(255,56,96)',   'rgba(255,56,96,0.12)');
      drawBboxes(c.good, 'rgb(35,209,96)',   'rgba(35,209,96,0.12)');
    }}

    wrap.addEventListener('mousemove', function(e) {{
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left; mouseY = e.clientY - rect.top;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(draw);
    }});
    wrap.addEventListener('mouseleave', function() {{
      mouseX = null; mouseY = null;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(draw);
    }});
    new ResizeObserver(resize).observe(wrap);
    resize();
  }});
}})();
</script>
</body>
</html>
"""


def write_selected_annotations(cards: list[dict], out_json: Path) -> None:
  """Write selected interactive entries for index.html consumption."""
  payload = [
    {
      "bg_id": c["bg_id"],
      "fg_class": c["fg_class"],
      "combo_idx": c["combo_idx"],
      "image_file": f"data:image/jpeg;base64,{c['img_b64']}",
      "img_size": IMG_SIZE,
      "good_bboxes": c["good_bboxes"],
      "bad_bboxes": c["bad_bboxes"],
    }
    for c in cards
  ]
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(payload, indent=2))



def main():
    parser = argparse.ArgumentParser(description="Generate bbox_reveal_viz.html")
    parser.add_argument("--out", default=str(Path(__file__).parent / "bbox_reveal_viz.html"),
                        help="Output HTML path")
  parser.add_argument(
    "--out-json",
    default=str(Path(__file__).parent / "data" / "annotations_selected.json"),
    help="Output selected annotations JSON for index interactive section",
  )
    args = parser.parse_args()

    print("Loading detection data...")
    detections, bg_combo_map = load_detections()

    if CHOSEN is None:
        print("Discovering all combos...")
        chosen = discover_all_combos(detections, bg_combo_map)
    else:
        chosen = CHOSEN

    print(f"Building card data for {len(chosen)} combos...")
    cards = build_card_data(detections, bg_combo_map, chosen)

    print("Rendering HTML...")
    html = build_html(cards)

    out_path = Path(args.out)
    out_path.write_text(html)
    print(f"Written: {out_path}  ({len(html)//1024} KB)")

    out_json = Path(args.out_json)
    write_selected_annotations(cards, out_json)
    print(f"Written: {out_json}")


if __name__ == "__main__":
    main()
