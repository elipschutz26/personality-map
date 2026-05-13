import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from data_loader import load_scores, load_character_names, load_bap_labels

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
        hovertemplate=(
            "<b>%{text}</b><br>"
            + src + "<br>"
            "PC1 (villain↔hero): %{x:.1f}<br>"
            "PC2 (goofy↔serious): %{y:.1f}<br>"
            "PC3 (cool↔dorky): %{z:.1f}"
            "<extra></extra>"
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
    title=f"Personality Map 3D — {len(scores)} characters via PCA "
          f"(PC1+PC2+PC3 = {ev.sum():.1f}% variance)",
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
)

datalist_options = "\n    ".join(
    f'<option value="{n}"></option>' for n in sorted(name_to_id.keys())
)
trait_options = "\n    ".join(
    f'<option value="{a}"></option>' for a in sorted(adjective_to_chars.keys())
)
plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Personality Map 3D</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 0; background: white; }}
  .controls {{ padding: 12px 16px; background: #f6f6f6; border-bottom: 1px solid #ddd;
              display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  input#search {{ padding: 7px 10px; font-size: 14px; min-width: 320px;
                  border: 1px solid #bbb; border-radius: 4px; }}
  button {{ padding: 7px 14px; font-size: 14px; cursor: pointer;
            border: 1px solid #bbb; background: white; border-radius: 4px; }}
  button:hover {{ background: #eee; }}
  .hint {{ color: #555; font-size: 13px; }}
  .sep {{ color: #aaa; font-size: 13px; }}
  #status b {{ color: #c0392b; }}
</style>
</head>
<body>
<div class="controls">
  <label for="search"><b>Focus on character:</b></label>
  <input id="search" list="charlist"
         placeholder="e.g. Hermione Granger, Walter White">
  <datalist id="charlist">
    {datalist_options}
  </datalist>
  <span class="sep">or</span>
  <label for="traitsearch"><b>Most extreme on trait:</b></label>
  <input id="traitsearch" list="traitlist"
         placeholder="e.g. kind, evil, smart, lazy">
  <datalist id="traitlist">
    {trait_options}
  </datalist>
  <button id="reset">Show all</button>
  <div class="hint" id="status" style="flex-basis:100%; margin-top:6px;">
    All {len(scores)} characters shown. Pick a character to see top 10 similar + top 10 opposite, or pick a trait to see the 10 most extreme.
  </div>
</div>
{plot_div}
<script>
const traceCharIds = {json.dumps(trace_char_ids)};
const traceDefaultSize = {json.dumps(trace_default_size)};
const showTraceIndices = {json.dumps(show_trace_indices)};
const simData = {json.dumps(sim_data)};
const idToName = {json.dumps(id_to_name)};
const nameToId = {json.dumps(name_to_id)};
const adjectiveToChars = {json.dumps(adjective_to_chars)};

function applyFilter(focalId) {{
  const focalName = idToName[focalId] || focalId;
  const close = simData[focalId].close;
  const far = simData[focalId].far;
  const visible = new Set([focalId, ...close, ...far]);

  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const cids = traceCharIds[k];
    const sizes = cids.map(cid =>
      cid === focalId ? 12 : visible.has(cid) ? 6 : 0
    );
    Plotly.restyle('plot', {{'marker.size': [sizes]}}, [idx]);
  }}
  const closeNames = close.slice(0, 5).map(cid => idToName[cid] || cid).join(', ');
  const farNames = far.slice(0, 3).map(cid => idToName[cid] || cid).join(', ');
  document.getElementById('status').innerHTML =
    `Focused on <b>${{focalName}}</b> — closest: ${{closeNames}}… &nbsp;|&nbsp; most opposite: ${{farNames}}…`;
}}

function resetView() {{
  for (let k = 0; k < showTraceIndices.length; k++) {{
    const idx = showTraceIndices[k];
    const cids = traceCharIds[k];
    const sizes = cids.map(_ => traceDefaultSize[k]);
    Plotly.restyle('plot', {{'marker.size': [sizes]}}, [idx]);
  }}
  document.getElementById('status').textContent =
    `All {len(scores)} characters shown. Pick a name above to focus on its 10 closest and 10 most-opposite characters.`;
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
  const orderedNames = focal.map(cid => idToName[cid] || cid).join(', ');
  document.getElementById('status').innerHTML =
    `Top 10 most <b>${{adjective}}</b> (most extreme first): ${{orderedNames}}`;
}}

document.getElementById('search').addEventListener('change', (e) => {{
  const name = e.target.value.trim();
  const id = nameToId[name];
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

with open("similarity_map_3d.html", "w") as f:
    f.write(html)

print("wrote similarity_map_3d.html with focus search")
print(f"top 3 PCs explain {ev.sum():.1f}% of variance ({ev[0]:.1f} / {ev[1]:.1f} / {ev[2]:.1f})")
