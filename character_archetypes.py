"""
character_archetypes.py
Cluster all characters into 8 personality archetypes using K-means,
then display on a 2D UMAP colored by archetype.
Click an archetype to see all its characters + portraits in a sidebar.
"""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
            img_paths[cid] = img_file.as_uri()

# Standardize then cluster
scaler = StandardScaler()
X = scaler.fit_transform(scores.values)

print("Running K-means...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X)

# Name clusters by their 3 most extreme traits (furthest from mean in that cluster)
global_mean = scores.mean().values
cluster_names = []
cluster_trait_summaries = []
bap_cols = scores.columns.tolist()

# Use only the first 88 BAPs for cluster naming (classic, interpretable labels)
CLEAN_BAPS = {b for b in bap_cols if int(b[3:]) <= 88}

for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    centroid = scores.values[mask].mean(axis=0)
    diffs = centroid - global_mean

    # Find most extreme traits restricted to clean BAPs
    clean_indices = [i for i, b in enumerate(bap_cols) if b in CLEAN_BAPS]
    clean_diffs = [(i, diffs[i]) for i in clean_indices]
    clean_diffs_sorted = sorted(clean_diffs, key=lambda x: -x[1])  # most above avg first

    trait_words = []
    for idx, diff in clean_diffs_sorted[:6]:
        bap = bap_cols[idx]
        val = centroid[idx]
        low_w, high_w = labels[bap]
        # Pick the word that describes this cluster's direction
        word = high_w if val > 50 else low_w
        if word not in trait_words:
            trait_words.append(word)
        if len(trait_words) == 2:
            break

    name = " · ".join(trait_words[:2]) if trait_words else f"Cluster {c+1}"
    cluster_names.append(name)

    # Full summary: top 6 most extreme clean traits for sidebar
    summary_parts = []
    for idx, diff in sorted(clean_diffs, key=lambda x: -abs(x[1]))[:6]:
        bap = bap_cols[idx]
        val = centroid[idx]
        low_w, high_w = labels[bap]
        word = high_w if val > 50 else low_w
        summary_parts.append(f"{word} ({val:.0f})")
    cluster_trait_summaries.append(summary_parts)

print("Running UMAP (this may take ~30 seconds)...")
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
xy = umap.fit_transform(scores.values)

# Build per-cluster character lists
cluster_chars = {}
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    cids = scores.index[mask].tolist()
    cluster_chars[c] = {
        "cids": cids,
        "name": cluster_names[c],
        "traits": cluster_trait_summaries[c],
        "count": int(mask.sum()),
    }

# Plotly colors
palette = px.colors.qualitative.Set2
color_map = {c: palette[c % len(palette)] for c in range(N_CLUSTERS)}

fig = go.Figure()
trace_cluster_ids = []
trace_default_sizes = []
show_trace_indices = []

for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    cids_in_trace = scores.index[mask].tolist()
    trace_cluster_ids.append(cids_in_trace)
    trace_default_sizes.append(5)
    show_trace_indices.append(c)
    fig.add_trace(go.Scatter(
        x=xy[mask, 0], y=xy[mask, 1],
        mode="markers",
        name=f"#{c+1}: {cluster_names[c]} ({mask.sum()})",
        marker=dict(size=5, color=color_map[c], opacity=0.75),
        text=[id_to_name[cid] for cid in cids_in_trace],
        customdata=cids_in_trace,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

fig.update_layout(
    title=f"Character Archetypes — {N_CLUSTERS} personality clusters via K-means + UMAP ({len(scores)} characters)",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=680,
    margin=dict(r=300),
    legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
    plot_bgcolor="#fafafa",
)

plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Character Archetypes</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 0; background: #f8f8f8; }}
  h1 {{ margin: 0; padding: 14px 20px 0; font-size: 22px; }}
  .subtitle {{ color: #666; padding: 4px 20px 12px; font-size: 14px; border-bottom: 1px solid #ddd; }}
  .main {{ display: flex; }}
  #plot-wrap {{ flex: 1; }}
  #sidebar {{ width: 280px; flex-shrink: 0; background: white; border-left: 1px solid #ddd;
              padding: 16px; overflow-y: auto; max-height: 100vh; }}
  #sidebar h3 {{ margin: 0 0 4px; font-size: 16px; }}
  .cluster-tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 12px; color: white; margin-bottom: 8px; }}
  .trait-pill {{ display: inline-block; padding: 3px 8px; margin: 3px 3px 0 0;
                 border-radius: 12px; background: #eef; font-size: 12px; color: #338; }}
  .char-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 12px; }}
  .char-thumb {{ text-align: center; }}
  .char-thumb img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px;
                     background: #eee; display: block; }}
  .char-thumb span {{ font-size: 10px; color: #555; display: block; margin-top: 2px;
                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .empty-state {{ color: #aaa; font-size: 13px; text-align: center; margin-top: 40px; padding: 20px; }}
  .archetype-list {{ margin-top: 8px; }}
  .arch-item {{ padding: 10px 12px; border-radius: 6px; margin-bottom: 6px; cursor: pointer;
                border: 1px solid transparent; transition: all 0.1s; }}
  .arch-item:hover {{ border-color: #bbb; background: #f5f5f5; }}
  .arch-item.active {{ border-color: #666; background: #f0f0f0; }}
  .arch-name {{ font-weight: 600; font-size: 13px; }}
  .arch-count {{ font-size: 12px; color: #888; }}
</style>
</head>
<body>
<h1>Character Personality Archetypes</h1>
<p class="subtitle">{len(scores)} characters clustered into {N_CLUSTERS} archetypes using K-means on 500 personality traits, then arranged by UMAP</p>
<div class="main">
  <div id="plot-wrap">{plot_div}</div>
  <div id="sidebar">
    <b style="font-size:14px">Click a dot or archetype:</b>
    <div class="archetype-list" id="arch-list"></div>
    <div id="detail"></div>
  </div>
</div>
<script>
const clusterChars = {json.dumps(cluster_chars)};
const idToName = {json.dumps(id_to_name)};
const imgPaths = {json.dumps(img_paths)};
const colorMap = {json.dumps({str(c): color_map[c] for c in range(N_CLUSTERS)})};
const traceClusterIds = {json.dumps(trace_cluster_ids)};
const traceDefaultSizes = {json.dumps(trace_default_sizes)};
const showTraceIndices = {json.dumps(show_trace_indices)};

// Build archetype list
const archList = document.getElementById('arch-list');
for (let c = 0; c < {N_CLUSTERS}; c++) {{
  const info = clusterChars[c];
  const div = document.createElement('div');
  div.className = 'arch-item';
  div.id = `arch-${{c}}`;
  div.style.borderLeft = `4px solid ${{colorMap[c]}}`;
  div.innerHTML = `
    <div class="arch-name">#${{c+1}}: ${{info.name}}</div>
    <div class="arch-count">${{info.count}} characters</div>`;
  div.addEventListener('click', () => showCluster(c));
  archList.appendChild(div);
}}

function showCluster(c) {{
  // Highlight active
  document.querySelectorAll('.arch-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`arch-${{c}}`).classList.add('active');

  const info = clusterChars[c];
  const traitPills = info.traits.map(t => `<span class="trait-pill">${{t}}</span>`).join('');

  // Show top 20 character portraits
  const topChars = info.cids.slice(0, 20);
  const thumbs = topChars.map(cid => {{
    const nm = idToName[cid] || cid;
    const img = imgPaths[cid];
    const imgEl = img
      ? `<img src="${{img}}" alt="${{nm}}" onerror="this.style.display='none'">`
      : `<div style="width:100%;aspect-ratio:1;background:#ddd;border-radius:6px;"></div>`;
    return `<div class="char-thumb">${{imgEl}}<span>${{nm}}</span></div>`;
  }}).join('');

  document.getElementById('detail').innerHTML = `
    <h3 style="margin-top:16px">Archetype #${{c+1}}: ${{info.name}}</h3>
    <div style="margin-bottom:8px">${{traitPills}}</div>
    <div style="font-size:12px;color:#888;margin-bottom:8px">${{info.count}} characters total</div>
    <div class="char-grid">${{thumbs}}</div>`;

  // Highlight this cluster in the plot
  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const isThis = (k === c);
    const size = isThis ? 8 : 3;
    const opacity = isThis ? 0.9 : 0.2;
    Plotly.restyle('plot', {{'marker.size': size, 'marker.opacity': opacity}}, [idx]);
  }}
}}

// Click on a dot → show that character's cluster
document.getElementById('plot').on('plotly_click', (data) => {{
  const pt = data.points[0];
  const cid = pt.customdata;
  if (!cid) return;
  // Find which cluster
  for (let c = 0; c < {N_CLUSTERS}; c++) {{
    if (clusterChars[c].cids.includes(cid)) {{
      showCluster(c);
      break;
    }}
  }}
}});
</script>
</body>
</html>"""

with open("character_archetypes.html", "w") as f:
    f.write(html)

print("wrote character_archetypes.html")
print(f"\nArchetype summary:")
for c in range(N_CLUSTERS):
    info = cluster_chars[c]
    print(f"  #{c+1}: {info['name']} — {info['count']} characters")
