"""
character_portrait_map.py
3D personality map with character portrait images shown on hover.
Hover over any dot to see the character's photo + top personality traits in the sidebar.
"""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from data_loader import load_scores, load_character_names, load_bap_labels

PICS_ROOT = (
    Path.home()
    / "Desktop/openpsychometrics"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "resources/pics"
)
TOP_N = 10

scores = load_scores()
names = load_character_names()
labels = load_bap_labels()
sources = scores.index.to_series().str.split("/").str[0]

pca = PCA(n_components=3, random_state=0)
xyz = pca.fit_transform(scores.values)

dist_matrix = pairwise_distances(scores.values)
char_ids_all = scores.index.tolist()
sim_data = {}
for i, cid in enumerate(char_ids_all):
    d = dist_matrix[i].copy()
    d[i] = np.inf
    sorted_idx = np.argsort(d)
    sim_data[cid] = {
        "close": [char_ids_all[j] for j in sorted_idx[:TOP_N]],
        "far":   [char_ids_all[j] for j in sorted_idx[::-1][:TOP_N]],
    }

id_to_name = {}
name_to_id = {}
for cid in scores.index:
    nm = names.loc[cid, "name"] if cid in names.index else cid
    id_to_name[cid] = nm
    if nm not in name_to_id:
        name_to_id[nm] = cid

display_names = [id_to_name[cid] for cid in scores.index]

# Build image path map (file:// URLs for local HTML)
img_paths = {}
for cid in scores.index:
    parts = cid.split("/")
    if len(parts) == 2:
        src, num = parts
        img_file = PICS_ROOT / src / f"{num}.jpg"
        if img_file.exists():
            img_paths[cid] = img_file.as_uri()

# Precompute top 5 most extreme traits for each character
# "extreme" = furthest from 50 (neutral midpoint)
bap_cols = scores.columns.tolist()
char_traits = {}
for cid in scores.index:
    row = scores.loc[cid]
    extremeness = (row - 50).abs()
    top5 = extremeness.nlargest(5).index.tolist()
    trait_strs = []
    for bap in top5:
        val = row[bap]
        low_word, high_word = labels[bap]
        word = high_word if val > 50 else low_word
        trait_strs.append(f"{word} ({val:.0f})")
    char_traits[cid] = trait_strs

top_sources = sources.value_counts().head(20).index.tolist()
display_source = sources.where(sources.isin(top_sources), "other")

palette = px.colors.qualitative.Light24
color_map = {src: palette[i % len(palette)] for i, src in enumerate(top_sources)}
color_map["other"] = "lightgray"

fig = go.Figure()

trace_char_ids: list[list[str]] = []
trace_default_size: list[int] = []
show_trace_indices: list[int] = []

trace_idx = 0
for src in top_sources + ["other"]:
    mask = (display_source == src).values
    if not mask.any():
        continue
    cids_in_trace = scores.index[mask].tolist()
    trace_char_ids.append(cids_in_trace)
    is_other = (src == "other")
    default_size = 3 if is_other else 4
    trace_default_size.append(default_size)
    show_trace_indices.append(trace_idx)
    fig.add_trace(go.Scatter3d(
        x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
        mode="markers",
        name=src,
        legendgroup=src,
        marker=dict(size=default_size,
                    color=color_map[src],
                    opacity=0.35 if is_other else 0.85),
        text=[display_names[i] for i in range(len(scores)) if mask[i]],
        customdata=[cids_in_trace[k] for k in range(len(cids_in_trace))],
        hovertemplate=(
            "<b>%{text}</b><br>"
            + src + "<extra></extra>"
        ),
    ))
    trace_idx += 1

ev = pca.explained_variance_ratio_ * 100
lim = max(abs(xyz.min()), abs(xyz.max()))
axis_line = dict(mode="lines", line=dict(color="black", width=3),
                 showlegend=False, hoverinfo="skip")
fig.add_trace(go.Scatter3d(x=[-lim, lim], y=[0, 0], z=[0, 0], legendgroup="_chrome", **axis_line))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[-lim, lim], z=[0, 0], legendgroup="_chrome", **axis_line))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-lim, lim], legendgroup="_chrome", **axis_line))

label_off = lim * 1.08
pole_annotations = [
    dict(x=-label_off, y=0, z=0, text=f"<b>villain</b><br>PC1 ({ev[0]:.1f}%)", showarrow=False, font=dict(size=14)),
    dict(x= label_off, y=0, z=0, text="<b>hero</b>", showarrow=False, font=dict(size=14)),
    dict(x=0, y=-label_off, z=0, text=f"<b>goofy</b><br>PC2 ({ev[1]:.1f}%)", showarrow=False, font=dict(size=14)),
    dict(x=0,  y=label_off, z=0, text="<b>serious</b>", showarrow=False, font=dict(size=14)),
    dict(x=0, y=0, z=-label_off, text=f"<b>cool</b><br>PC3 ({ev[2]:.1f}%)", showarrow=False, font=dict(size=14)),
    dict(x=0, y=0,  z=label_off, text="<b>dorky</b>", showarrow=False, font=dict(size=14)),
]

hidden_axis = dict(visible=False, showgrid=False, showbackground=False,
                   showline=False, showticklabels=False, zeroline=False)

fig.update_layout(
    title=f"Character Personality Map — {len(scores)} characters (hover for portrait)",
    scene=dict(
        xaxis=hidden_axis,
        yaxis=hidden_axis,
        zaxis=hidden_axis,
        aspectmode="cube",
        annotations=pole_annotations,
    ),
    legend=dict(itemsizing="constant",
                itemclick="toggleothers",
                itemdoubleclick="toggle"),
    height=720,
    margin=dict(r=270),  # leave room for sidebar
)

datalist_options = "\n    ".join(
    f'<option value="{n}"></option>' for n in sorted(name_to_id.keys())
)

def is_real_word(s: str) -> bool:
    return "&#" not in s and s.strip() != ""

adjective_to_chars: dict[str, list[str]] = {}
for bap_id, (low_word, high_word) in labels.items():
    if bap_id not in scores.columns:
        continue
    if not (is_real_word(low_word) and is_real_word(high_word)):
        continue
    col = scores[bap_id].values
    low_order = np.argsort(col)[:TOP_N]
    high_order = np.argsort(col)[::-1][:TOP_N]
    if low_word not in adjective_to_chars:
        adjective_to_chars[low_word] = [char_ids_all[i] for i in low_order]
    if high_word not in adjective_to_chars:
        adjective_to_chars[high_word] = [char_ids_all[i] for i in high_order]

trait_options = "\n    ".join(
    f'<option value="{a}"></option>' for a in sorted(adjective_to_chars.keys())
)

plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Character Portrait Map</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 0; background: #f8f8f8; display: flex; flex-direction: column; height: 100vh; }}
  .controls {{ padding: 10px 16px; background: #f0f0f0; border-bottom: 1px solid #ddd;
               display: flex; gap: 10px; align-items: center; flex-wrap: wrap; flex-shrink: 0; }}
  input {{ padding: 6px 10px; font-size: 14px; min-width: 260px;
           border: 1px solid #bbb; border-radius: 4px; }}
  button {{ padding: 6px 14px; font-size: 14px; cursor: pointer;
             border: 1px solid #bbb; background: white; border-radius: 4px; }}
  button:hover {{ background: #eee; }}
  .hint {{ color: #555; font-size: 13px; }}
  .sep {{ color: #aaa; }}
  #status b {{ color: #c0392b; }}
  .main {{ display: flex; flex: 1; overflow: hidden; }}
  #plot-wrap {{ flex: 1; }}
  #sidebar {{ width: 260px; flex-shrink: 0; background: white; border-left: 1px solid #ddd;
              padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }}
  #sidebar h3 {{ margin: 0; font-size: 16px; }}
  #sidebar .source-tag {{ font-size: 12px; color: #888; margin-top: 2px; }}
  #portrait-img {{ width: 100%; border-radius: 8px; object-fit: cover; background: #eee;
                   min-height: 120px; display: block; }}
  .no-img {{ width: 100%; height: 140px; background: #eee; border-radius: 8px;
             display: flex; align-items: center; justify-content: center; color: #aaa; font-size: 13px; }}
  .trait-list {{ list-style: none; margin: 0; padding: 0; }}
  .trait-list li {{ padding: 4px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
  .trait-list li:last-child {{ border-bottom: none; }}
  .trait-bar {{ display: inline-block; height: 8px; background: #3498db; border-radius: 4px;
                margin-right: 6px; vertical-align: middle; }}
  .similar-list {{ font-size: 12px; color: #555; line-height: 1.6; }}
  .empty-state {{ color: #aaa; font-size: 13px; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>
<div class="controls">
  <label for="search"><b>Find character:</b></label>
  <input id="search" list="charlist" placeholder="e.g. Hermione Granger, Walter White">
  <datalist id="charlist">{datalist_options}</datalist>
  <span class="sep">or</span>
  <label for="traitsearch"><b>Trait extreme:</b></label>
  <input id="traitsearch" list="traitlist" placeholder="e.g. brave, villainous, genius">
  <datalist id="traitlist">{trait_options}</datalist>
  <button id="reset">Show all</button>
  <div class="hint" id="status" style="flex-basis:100%;">
    {len(scores)} characters shown. Hover over a dot to see the portrait, or search above.
  </div>
</div>
<div class="main">
  <div id="plot-wrap">{plot_div}</div>
  <div id="sidebar">
    <div class="empty-state">Hover over a character<br>to see their portrait<br>and personality traits</div>
  </div>
</div>
<script>
const traceCharIds = {json.dumps(trace_char_ids)};
const traceDefaultSize = {json.dumps(trace_default_size)};
const showTraceIndices = {json.dumps(show_trace_indices)};
const simData = {json.dumps(sim_data)};
const idToName = {json.dumps(id_to_name)};
const nameToId = {json.dumps(name_to_id)};
const adjectiveToChars = {json.dumps(adjective_to_chars)};
const imgPaths = {json.dumps(img_paths)};
const charTraits = {json.dumps(char_traits)};

// Map char_id -> source name (first part before /)
function getSource(cid) {{ return cid.split('/')[0]; }}

function renderSidebar(cid) {{
  const name = idToName[cid] || cid;
  const src = getSource(cid);
  const img = imgPaths[cid];
  const traits = charTraits[cid] || [];
  const close = (simData[cid] || {{}}).close || [];

  const imgHtml = img
    ? `<img id="portrait-img" src="${{img}}" alt="${{name}}" onerror="this.style.display='none'">`
    : `<div class="no-img">No image</div>`;

  const traitItems = traits.map(t => {{
    const match = t.match(/(.+) \\((\\d+)\\)/);
    if (!match) return `<li>${{t}}</li>`;
    const word = match[1], val = parseInt(match[2]);
    const barW = Math.round((Math.abs(val - 50) / 50) * 80);
    return `<li><span class="trait-bar" style="width:${{barW}}px"></span>${{word}} <span style="color:#999">(${{val}})</span></li>`;
  }}).join('');

  const closeNames = close.slice(0, 5).map(c => idToName[c] || c).join(', ');

  document.getElementById('sidebar').innerHTML = `
    <h3>${{name}}</h3>
    <div class="source-tag">${{src}}</div>
    ${{imgHtml}}
    <div><b style="font-size:13px">Top traits:</b>
      <ul class="trait-list">${{traitItems}}</ul>
    </div>
    <div><b style="font-size:13px">Most similar to:</b>
      <div class="similar-list">${{closeNames}}</div>
    </div>`;
}}

function applyFilter(focalId) {{
  const focalName = idToName[focalId] || focalId;
  const close = simData[focalId].close;
  const far = simData[focalId].far;
  const visible = new Set([focalId, ...close, ...far]);

  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const cids = traceCharIds[k];
    const sizes = cids.map(cid => cid === focalId ? 12 : visible.has(cid) ? 6 : 0);
    Plotly.restyle('plot', {{'marker.size': [sizes]}}, [idx]);
  }}
  renderSidebar(focalId);
  const closeNames = close.slice(0, 5).map(cid => idToName[cid] || cid).join(', ');
  document.getElementById('status').innerHTML =
    `Focused on <b>${{focalName}}</b> — 10 closest + 10 most-opposite shown`;
}}

function resetView() {{
  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const cids = traceCharIds[k];
    const sizes = cids.map(_ => traceDefaultSize[k]);
    Plotly.restyle('plot', {{'marker.size': [sizes]}}, [idx]);
  }}
  document.getElementById('status').textContent =
    `{len(scores)} characters shown. Hover over a dot to see the portrait, or search above.`;
  document.getElementById('sidebar').innerHTML =
    '<div class="empty-state">Hover over a character<br>to see their portrait<br>and personality traits</div>';
}}

function applyTraitFilter(adjective) {{
  const focal = adjectiveToChars[adjective];
  if (!focal) return;
  const visible = new Set(focal);
  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const cids = traceCharIds[k];
    const sizes = cids.map(cid => visible.has(cid) ? 8 : 0);
    Plotly.restyle('plot', {{'marker.size': [sizes]}}, [idx]);
  }}
  document.getElementById('status').innerHTML =
    `Top 10 most <b>${{adjective}}</b>: ${{focal.map(cid => idToName[cid] || cid).join(', ')}}`;
  if (focal.length > 0) renderSidebar(focal[0]);
}}

// Hover event → update sidebar
document.getElementById('plot').on('plotly_hover', (data) => {{
  const pt = data.points[0];
  const cid = pt.customdata;
  if (cid) renderSidebar(cid);
}});

document.getElementById('search').addEventListener('change', (e) => {{
  const id = nameToId[e.target.value.trim()];
  if (id) {{
    document.getElementById('traitsearch').value = '';
    applyFilter(id);
  }}
}});

document.getElementById('traitsearch').addEventListener('change', (e) => {{
  const adj = e.target.value.trim().toLowerCase();
  if (adjectiveToChars[adj]) {{
    document.getElementById('search').value = '';
    applyTraitFilter(adj);
  }}
}});

document.getElementById('reset').addEventListener('click', () => {{
  document.getElementById('search').value = '';
  document.getElementById('traitsearch').value = '';
  resetView();
}});
</script>
</body>
</html>"""

with open("character_portrait_map.html", "w") as f:
    f.write(html)

print("wrote character_portrait_map.html")
print(f"  {len(scores)} characters, {len(img_paths)} with portrait images")
print(f"  PCA variance explained: PC1={ev[0]:.1f}% PC2={ev[1]:.1f}% PC3={ev[2]:.1f}%")
