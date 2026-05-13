import umap
import plotly.graph_objects as go
import plotly.express as px
from data_loader import load_scores, load_character_names

HIGHLIGHT_ID = "HP/3"

scores = load_scores()
names = load_character_names()
sources = scores.index.to_series().str.split("/").str[0]

reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=0)
xyz = reducer.fit_transform(scores.values)

display_names = [
    names.loc[cid, "name"] if cid in names.index else cid
    for cid in scores.index
]

top_sources = sources.value_counts().head(20).index.tolist()
display_source = sources.where(sources.isin(top_sources), "other")

palette = px.colors.qualitative.Light24
color_map = {src: palette[i % len(palette)] for i, src in enumerate(top_sources)}
color_map["other"] = "lightgray"

fig = go.Figure()

for src in top_sources + ["other"]:
    mask = (display_source == src).values
    if not mask.any():
        continue
    fig.add_trace(go.Scatter3d(
        x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
        mode="markers",
        name=src,
        marker=dict(size=3 if src == "other" else 4,
                    color=color_map[src],
                    opacity=0.35 if src == "other" else 0.85),
        text=[display_names[i] for i in range(len(scores)) if mask[i]],
        hovertemplate="<b>%{text}</b><br>" + src + "<extra></extra>",
    ))

if HIGHLIGHT_ID in scores.index:
    idx = scores.index.get_loc(HIGHLIGHT_ID)
    name = names.loc[HIGHLIGHT_ID, "name"] if HIGHLIGHT_ID in names.index else HIGHLIGHT_ID
    fig.add_trace(go.Scatter3d(
        x=[xyz[idx, 0]], y=[xyz[idx, 1]], z=[xyz[idx, 2]],
        mode="markers+text",
        name=name,
        marker=dict(size=10, color="red", symbol="diamond",
                    line=dict(color="black", width=2)),
        text=[name],
        textposition="top center",
        textfont=dict(size=14, color="black"),
        hovertemplate=f"<b>{name}</b><extra></extra>",
    ))

fig.update_layout(
    title=f"Personality Map 3D (UMAP) — {len(scores)} characters<br>"
          f"<sub>UMAP axes have no inherent meaning — they're optimized to keep similar characters near each other</sub>",
    scene=dict(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        zaxis_title="UMAP-3",
        aspectmode="cube",
    ),
    legend=dict(itemsizing="constant"),
    height=850,
)

out = "similarity_map_umap.html"
fig.write_html(out, include_plotlyjs="cdn")
print(f"wrote {out}")
