"""
gestalt_map.py
Gestalt personality map — character portrait images placed directly at their
UMAP positions. Clusters emerge visually from the faces themselves.
Zoom with scroll wheel, pan by dragging, hover for details.
"""
import json
from pathlib import Path
import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from data_loader import load_scores, load_character_names, load_bap_labels

PICS_ROOT = (
    Path.home()
    / "Desktop/openpsychometrics"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "resources/pics"
)
N_CLUSTERS = 8

scores = load_scores()
names = load_character_names()
labels = load_bap_labels()

id_to_name = {cid: (names.loc[cid, "name"] if cid in names.index else cid) for cid in scores.index}

# Image paths
img_paths = {}
for cid in scores.index:
    parts = cid.split("/")
    if len(parts) == 2:
        src, num = parts
        img_file = PICS_ROOT / src / f"{num}.jpg"
        if img_file.exists():
            img_paths[cid] = f"pics/{src}/{num}.jpg"

print("Running K-means clustering...")
scaler = StandardScaler()
X = scaler.fit_transform(scores.values)
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels_arr = km.fit_predict(X)

# Name clusters (using clean BAPs ≤ 88)
global_mean = scores.mean().values
bap_cols = scores.columns.tolist()
CLEAN_BAPS = {b for b in bap_cols if int(b[3:]) <= 88}
cluster_names = []
cluster_trait_summaries = []

for c in range(N_CLUSTERS):
    mask = cluster_labels_arr == c
    centroid = scores.values[mask].mean(axis=0)
    diffs = centroid - global_mean
    clean_indices = [i for i, b in enumerate(bap_cols) if b in CLEAN_BAPS]
    clean_diffs = sorted([(i, diffs[i]) for i in clean_indices], key=lambda x: -x[1])
    trait_words = []
    for idx, diff in clean_diffs[:8]:
        bap = bap_cols[idx]
        val = centroid[idx]
        low_w, high_w = labels[bap]
        word = high_w if val > 50 else low_w
        if word not in trait_words:
            trait_words.append(word)
        if len(trait_words) == 2:
            break
    cluster_names.append(" · ".join(trait_words[:2]) or f"Cluster {c+1}")
    summary = []
    for idx, diff in sorted([(i, diffs[i]) for i in clean_indices], key=lambda x: -abs(x[1]))[:5]:
        bap = bap_cols[idx]
        val = centroid[idx]
        low_w, high_w = labels[bap]
        word = high_w if val > 50 else low_w
        summary.append(f"{word} ({val:.0f})")
    cluster_trait_summaries.append(summary)

print("Running UMAP (this may take ~30 seconds)...")
umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.05, random_state=42)
xy = umap.fit_transform(scores.values)

# Push cluster centroids apart so there are clear gaps between archetypes
centroids = np.array([xy[cluster_labels_arr == c].mean(axis=0) for c in range(N_CLUSTERS)])
global_centroid = centroids.mean(axis=0)
spread_factor = 2.8  # how far apart to push clusters
new_centroids = global_centroid + (centroids - global_centroid) * spread_factor
xy_spread = xy.copy()
for c in range(N_CLUSTERS):
    mask = cluster_labels_arr == c
    shift = new_centroids[c] - centroids[c]
    xy_spread[mask] += shift

# Normalize to 0-1 range with padding
pad = 0.04
xy_min = xy_spread.min(axis=0)
xy_max = xy_spread.max(axis=0)
xy_norm = (xy_spread - xy_min) / (xy_max - xy_min) * (1 - 2 * pad) + pad

# Cluster colors
palette = px.colors.qualitative.Set2
cluster_colors = [palette[c % len(palette)] for c in range(N_CLUSTERS)]

# Build per-character data list for JS
chars_data = []
char_ids = scores.index.tolist()
for i, cid in enumerate(char_ids):
    chars_data.append({
        "id": cid,
        "name": id_to_name.get(cid, cid),
        "src": cid.split("/")[0],
        "cluster": int(cluster_labels_arr[i]),
        "x": float(xy_norm[i, 0]),
        "y": float(xy_norm[i, 1]),
        "img": img_paths.get(cid, ""),
    })

cluster_info = [
    {"name": cluster_names[c], "color": cluster_colors[c],
     "traits": cluster_trait_summaries[c], "count": int((cluster_labels_arr == c).sum())}
    for c in range(N_CLUSTERS)
]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Character Gestalt Map</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #111; color: #eee; }}
  #topbar {{ padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 16px; flex-shrink: 0; z-index: 10; }}
  #topbar h1 {{ font-size: 16px; font-weight: 600; white-space: nowrap; }}
  #topbar .hint {{ font-size: 12px; color: #888; }}
  #legend {{ display: flex; gap: 10px; flex-wrap: wrap; flex: 1; justify-content: flex-end; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px;
                  cursor: pointer; padding: 3px 7px; border-radius: 12px;
                  border: 1px solid transparent; transition: all 0.15s; }}
  .legend-item:hover {{ background: rgba(255,255,255,0.08); }}
  .legend-item.dimmed {{ opacity: 0.35; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  #canvas-wrap {{ height: calc(100vh - 52px); position: relative; overflow: hidden; cursor: grab; }}
  #canvas-wrap.dragging {{ cursor: grabbing; }}
  canvas {{ position: absolute; top: 0; left: 0; }}
  #tooltip {{ position: fixed; background: rgba(20,20,20,0.95); border: 1px solid #444;
              border-radius: 8px; padding: 0; pointer-events: none; display: none;
              z-index: 100; box-shadow: 0 4px 20px rgba(0,0,0,0.5); max-width: 220px; }}
  #tooltip img {{ width: 220px; height: 160px; object-fit: cover; border-radius: 7px 7px 0 0; display: block; }}
  #tooltip .tt-body {{ padding: 10px 12px; }}
  #tooltip .tt-name {{ font-weight: 600; font-size: 14px; }}
  #tooltip .tt-src {{ font-size: 11px; color: #888; margin-top: 2px; }}
  #tooltip .tt-traits {{ font-size: 11px; color: #aaa; margin-top: 6px; line-height: 1.5; }}
  #search-wrap {{ position: absolute; top: 12px; left: 12px; z-index: 20; }}
  #search {{ padding: 7px 12px; font-size: 13px; background: #1a1a1a; color: #eee;
             border: 1px solid #444; border-radius: 6px; width: 220px; }}
  #search:focus {{ outline: none; border-color: #888; }}
  #search-results {{ background: #1a1a1a; border: 1px solid #444; border-radius: 6px;
                     margin-top: 4px; max-height: 200px; overflow-y: auto; display: none; }}
  .search-result {{ padding: 7px 12px; font-size: 13px; cursor: pointer; }}
  .search-result:hover {{ background: #2a2a2a; }}
  #zoom-hint {{ position: absolute; bottom: 14px; right: 16px; font-size: 12px; color: #555; z-index: 20; }}
</style>
</head>
<body>
<div id="topbar">
  <h1>Character Gestalt Map</h1>
  <span class="hint">Scroll to zoom · Drag to pan · Hover for details</span>
  <div id="legend"></div>
</div>
<div id="canvas-wrap">
  <canvas id="map"></canvas>
  <div id="search-wrap">
    <input id="search" placeholder="Find a character…" autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="zoom-hint">Scroll to zoom</div>
</div>
<div id="tooltip">
  <img id="tt-img" src="" alt="">
  <div class="tt-body">
    <div class="tt-name" id="tt-name"></div>
    <div class="tt-src" id="tt-src"></div>
    <div class="tt-traits" id="tt-traits"></div>
  </div>
</div>

<script>
const chars = {json.dumps(chars_data)};
const clusterInfo = {json.dumps(cluster_info)};
const N = chars.length;

// Preload images
const images = new Array(N);
let loadedCount = 0;
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');

// Transform state
let zoom = 1, panX = 0, panY = 0;
let isDragging = false, dragStart = {{x: 0, y: 0}}, panStart = {{x: 0, y: 0}};
const IMG_SIZE = 36;  // px at zoom=1
let highlightedCluster = -1;  // -1 = show all
let spotlightIdx = -1;

// Build legend
const legend = document.getElementById('legend');
clusterInfo.forEach((ci, c) => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.id = `leg-${{c}}`;
  item.innerHTML = `<div class="legend-dot" style="background:${{ci.color}}"></div>
    <span>#${{c+1}} ${{ci.name}} (${{ci.count}})</span>`;
  item.addEventListener('click', () => {{
    if (highlightedCluster === c) highlightedCluster = -1;
    else highlightedCluster = c;
    updateLegendState();
    drawMap();
  }});
  legend.appendChild(item);
}});

function updateLegendState() {{
  document.querySelectorAll('.legend-item').forEach((el, c) => {{
    if (highlightedCluster === -1) el.classList.remove('dimmed');
    else el.classList.toggle('dimmed', c !== highlightedCluster);
  }});
}}

function resize() {{
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  drawMap();
}}

// World → screen
function worldToScreen(wx, wy) {{
  const w = canvas.width, h = canvas.height;
  const sx = wx * w * zoom + panX;
  const sy = wy * h * zoom + panY;
  return {{x: sx, y: sy}};
}}
function screenToWorld(sx, sy) {{
  const w = canvas.width, h = canvas.height;
  return {{x: (sx - panX) / (w * zoom), y: (sy - panY) / (h * zoom)}};
}}

function drawMap() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Dark starfield bg
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const sz = IMG_SIZE * zoom;
  const half = sz / 2;

  // Draw chars back-to-front (dimmed first, then full)
  const order = [...Array(N).keys()].sort((a, b) => {{
    const aFull = highlightedCluster === -1 || chars[a].cluster === highlightedCluster;
    const bFull = highlightedCluster === -1 || chars[b].cluster === highlightedCluster;
    return aFull - bFull;
  }});

  for (const i of order) {{
    const c = chars[i];
    const {{x, y}} = worldToScreen(c.x, c.y);
    if (x < -half || x > canvas.width + half || y < -half || y > canvas.height + half) continue;

    const isFocused = highlightedCluster === -1 || c.cluster === highlightedCluster;
    const isSpotlit = spotlightIdx === i;

    ctx.save();
    ctx.globalAlpha = isFocused ? (isSpotlit ? 1 : 0.88) : 0.12;

    const img = images[i];
    if (img && img.complete && img.naturalWidth > 0) {{
      // Clip to circle
      ctx.beginPath();
      ctx.arc(x, y, half * 0.9, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(img, x - half, y - half, sz, sz);
    }} else {{
      // Fallback colored circle
      ctx.beginPath();
      ctx.arc(x, y, half * 0.9, 0, Math.PI * 2);
      ctx.fillStyle = clusterInfo[c.cluster].color;
      ctx.fill();
    }}

    // Colored ring
    ctx.restore();
    ctx.save();
    ctx.globalAlpha = isFocused ? 0.9 : 0.12;
    ctx.beginPath();
    ctx.arc(x, y, half * 0.9 + 1.5, 0, Math.PI * 2);
    ctx.strokeStyle = isSpotlit ? '#fff' : clusterInfo[c.cluster].color;
    ctx.lineWidth = isSpotlit ? 2.5 : 1.5;
    ctx.stroke();
    ctx.restore();
  }}
}}

// Lazy load images and redraw
for (let i = 0; i < N; i++) {{
  const c = chars[i];
  if (!c.img) {{ images[i] = null; continue; }}
  const img = new Image();
  img.src = c.img;
  img.onload = () => {{ loadedCount++; drawMap(); }};
  img.onerror = () => {{ images[i] = null; }};
  images[i] = img;
}}

// Zoom
canvas.parentElement.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const newZoom = Math.max(0.4, Math.min(20, zoom * zoomFactor));
  panX = mx - (mx - panX) * (newZoom / zoom);
  panY = my - (my - panY) * (newZoom / zoom);
  zoom = newZoom;
  drawMap();
}}, {{ passive: false }});

// Pan
canvas.parentElement.addEventListener('mousedown', (e) => {{
  if (e.button !== 0) return;
  isDragging = true;
  dragStart = {{x: e.clientX, y: e.clientY}};
  panStart = {{x: panX, y: panY}};
  canvas.parentElement.classList.add('dragging');
}});
window.addEventListener('mousemove', (e) => {{
  if (isDragging) {{
    panX = panStart.x + (e.clientX - dragStart.x);
    panY = panStart.y + (e.clientY - dragStart.y);
    drawMap();
  }}
}});
window.addEventListener('mouseup', () => {{
  isDragging = false;
  canvas.parentElement.classList.remove('dragging');
}});

// Tooltip on hover
const tooltip = document.getElementById('tooltip');
const ttImg = document.getElementById('tt-img');
const ttName = document.getElementById('tt-name');
const ttSrc = document.getElementById('tt-src');
const ttTraits = document.getElementById('tt-traits');

function findNearest(mx, my) {{
  const sz = IMG_SIZE * zoom;
  let best = null, bestDist = sz * 0.6;
  for (let i = 0; i < N; i++) {{
    const c = chars[i];
    const {{x, y}} = worldToScreen(c.x, c.y);
    const d = Math.hypot(mx - x, my - y);
    if (d < bestDist) {{ bestDist = d; best = i; }}
  }}
  return best;
}}

canvas.parentElement.addEventListener('mousemove', (e) => {{
  if (isDragging) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const i = findNearest(mx, my);

  if (i !== null && i !== spotlightIdx) {{
    spotlightIdx = i;
    drawMap();
    const c = chars[i];
    const ci = clusterInfo[c.cluster];
    ttName.textContent = c.name;
    ttSrc.textContent = `${{c.src}} · Archetype #${{c.cluster + 1}}: ${{ci.name}}`;
    ttTraits.textContent = ci.traits.join(' · ');
    if (c.img) {{
      ttImg.src = c.img;
      ttImg.style.display = 'block';
    }} else {{
      ttImg.style.display = 'none';
    }}
    // Position tooltip
    const tx = Math.min(e.clientX + 14, window.innerWidth - 240);
    const ty = Math.min(e.clientY - 20, window.innerHeight - 300);
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
    tooltip.style.display = 'block';
  }} else if (i === null) {{
    spotlightIdx = -1;
    tooltip.style.display = 'none';
    drawMap();
  }}
}});
canvas.parentElement.addEventListener('mouseleave', () => {{
  tooltip.style.display = 'none';
  spotlightIdx = -1;
  drawMap();
}});

// Search
const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');
const nameToIdx = {{}};
chars.forEach((c, i) => {{ nameToIdx[c.name.toLowerCase()] = i; }});

searchInput.addEventListener('input', () => {{
  const q = searchInput.value.toLowerCase().trim();
  if (q.length < 2) {{ searchResults.style.display = 'none'; return; }}
  const matches = chars.map((c, i) => ({{c, i}}))
    .filter(({{}}) => true)
    .reduce((acc, {{c, i}}) => {{
      if (c.name.toLowerCase().includes(q)) acc.push({{c, i}});
      return acc;
    }}, []).slice(0, 8);
  if (!matches.length) {{ searchResults.style.display = 'none'; return; }}
  searchResults.innerHTML = matches.map(({{}}) => '').join('');
  matches.forEach(({{c, i}}) => {{
    const div = document.createElement('div');
    div.className = 'search-result';
    div.textContent = c.name + ' (' + c.src + ')';
    div.addEventListener('click', () => {{
      flyTo(i);
      searchResults.style.display = 'none';
      searchInput.value = c.name;
    }});
    searchResults.appendChild(div);
  }});
  searchResults.style.display = 'block';
}});
document.addEventListener('click', (e) => {{
  if (!e.target.closest('#search-wrap')) searchResults.style.display = 'none';
}});

function flyTo(i) {{
  const c = chars[i];
  zoom = 5;
  const w = canvas.width, h = canvas.height;
  panX = w / 2 - c.x * w * zoom;
  panY = h / 2 - c.y * h * zoom;
  spotlightIdx = i;
  drawMap();
}}

// Defer resize until after layout is painted
requestAnimationFrame(() => requestAnimationFrame(resize));
window.addEventListener('resize', resize);
</script>

<section style="background:white;border-radius:10px;padding:20px 24px;margin:24px 30px 30px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;">
    <div style="grid-column:1/-1;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#999;margin-bottom:2px;">How to read this chart</div>
    <div><h3 style="font-size:13px;font-weight:700;color:#c87941;margin:0 0 6px">How it was made</h3><p style="font-size:13px;color:#555;line-height:1.7;margin:0">Each character was scored on 500 personality traits, then UMAP compressed those into 2D coordinates that preserve neighborhood structure — characters with similar personalities end up nearby. K-means (k=8) grouped them into 8 archetype clusters, each named by its most extreme traits.</p></div>
    <div><h3 style="font-size:13px;font-weight:700;color:#c87941;margin:0 0 6px">What you're looking at</h3><p style="font-size:13px;color:#555;line-height:1.7;margin:0">Every dot is a real character portrait placed at its UMAP position. The colored ring around each photo shows which of the 8 archetype clusters they belong to (see the legend). Clusters are spread apart so the groupings are easy to see. Dot size is uniform — proximity is what matters.</p></div>
    <div><h3 style="font-size:13px;font-weight:700;color:#c87941;margin:0 0 6px">How to interpret it</h3><p style="font-size:13px;color:#555;line-height:1.7;margin:0">Faces that appear next to each other share a personality profile, even if they're from completely different shows. Use the search bar to fly to any character. Click a cluster in the legend to isolate it. Hover any face to see their name, franchise, and archetype. Zoom and pan to explore dense regions.</p></div>
  </div>
</section>
</body>
</html>"""

with open("gestalt_map.html", "w") as f:
    f.write(html)

print("wrote gestalt_map.html")
print(f"  {len(chars_data)} characters, {len(img_paths)} with portraits")
print(f"\nArchetype clusters:")
for c, info in enumerate(cluster_info):
    print(f"  #{c+1}: {info['name']} ({info['count']} chars)")
