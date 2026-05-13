"""
franchise_profiles.py
Deep-dive into personality profiles per franchise.
Three panels:
  1. Franchise radar chart (compare two franchises on 12 key traits)
  2. Heatmap — top 20 franchises × 12 traits
  3. PCA scatter of franchise centroids
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Use plain JSON encoding so charts render in all browsers without bdata issues
pio.json.config.default_engine = "json"
from sklearn.decomposition import PCA
from data_loader import load_scores, load_character_names, load_bap_labels

scores = load_scores()
names = load_character_names()
labels = load_bap_labels()
sources = scores.index.to_series().str.split("/").str[0]

# 12 key interpretable traits (BAP id, display label, polarity)
# polarity: +1 means high score = "positive" end named in display
KEY_TRAITS = [
    ("BAP39", "Heroic"),       # heroic ↔ villainous (low=heroic)
    ("BAP84", "Kind"),         # cruel ↔ kind (high=kind)
    ("BAP28", "Loyal"),        # loyal ↔ traitorous (low=loyal)
    ("BAP26", "Wise"),         # wise ↔ foolish (low=wise)
    ("BAP2",  "Bold"),         # shy ↔ bold (high=bold)
    ("BAP19", "Brave"),        # brave ↔ careful (low=brave)
    ("BAP32", "Diligent"),     # diligent ↔ lazy (low=diligent)
    ("BAP37", "Genius"),       # dunce ↔ genius (high=genius)
    ("BAP1",  "Playful"),      # playful ↔ serious (low=playful)
    ("BAP29", "Creative"),     # creative ↔ conventional (low=creative)
    ("BAP35", "Emotional"),    # emotional ↔ logical (low=emotional)
    ("BAP25", "Forgiving"),    # forgiving ↔ vengeful (low=forgiving)
]

# For each trait, compute the "positive" score (some need to be flipped because
# the "good" pole is at the low end of the 0-100 scale for BAP39, 28, 26, 19, 32, 1, 29, 35, 25)
# We'll present them all as 0-100 where 100 = the named label
def adjusted_score(df_col, bap_id):
    """Return score oriented so 100 = the label in KEY_TRAITS display."""
    flip_ids = {"BAP39", "BAP28", "BAP26", "BAP19", "BAP32", "BAP1", "BAP29", "BAP35", "BAP25"}
    return 100 - df_col if bap_id in flip_ids else df_col

# Build per-character adjusted trait matrix
trait_cols = [t[0] for t in KEY_TRAITS]
trait_labels = [t[1] for t in KEY_TRAITS]
trait_df = pd.DataFrame(
    {label: adjusted_score(scores[bap], bap) for (bap, label) in KEY_TRAITS},
    index=scores.index
)

top_sources = sources.value_counts().head(25).index.tolist()
source_counts = sources.value_counts()

# Franchise centroids for each trait
franchise_means = {}
for src in top_sources:
    mask = sources == src
    franchise_means[src] = trait_df[mask].mean()

means_df = pd.DataFrame(franchise_means).T  # shape: (franchises, traits)

# ── Build franchise name lookup from known abbreviations ──
# We'll use the counts as franchise size; names are abbreviations
franchise_full_names = {
    "GOT": "Game of Thrones", "HP": "Harry Potter", "SW": "Star Wars",
    "LOTR": "Lord of the Rings", "BB": "Breaking Bad", "OF": "The Office",
    "FR": "Friends", "SCRUBS": "Scrubs", "S": "Seinfeld",
    "WD": "The Wire/Walking Dead", "BCS": "Better Call Saul",
    "MARVEL": "Marvel", "AH": "American Horror Story", "TB": "True Blood",
    "B99": "Brooklyn Nine-Nine", "PD": "Parks & Rec", "ATLA": "Avatar: ATLA",
    "HML": "Hannibal", "HAN": "Hannibal", "GA": "Grey's Anatomy",
}
def nice_name(src): return franchise_full_names.get(src, src)

# ── Panel 1: Interactive Radar chart (single franchise, dropdown) ──
radar_data = {}
for src in top_sources:
    radar_data[src] = means_df.loc[src].tolist()

# ── Panel 2: Heatmap ──
heatmap = go.Heatmap(
    z=means_df[trait_labels].values.tolist(),
    x=trait_labels,
    y=[nice_name(s) + f" ({source_counts[s]})" for s in top_sources],
    colorscale="RdYlGn",
    zmin=20, zmax=80,
    colorbar=dict(title="Score", thickness=12, x=1.01),
    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
)

# ── Panel 3: PCA scatter of franchise centroids ──
all_means = means_df[trait_labels].values
pca2 = PCA(n_components=2, random_state=0)
centroids_2d = pca2.fit_transform(all_means)
ev2 = pca2.explained_variance_ratio_ * 100

scatter = go.Scatter(
    x=centroids_2d[:, 0].tolist(),
    y=centroids_2d[:, 1].tolist(),
    mode="markers+text",
    text=[nice_name(s) for s in top_sources],
    textposition="top center",
    marker=dict(
        size=[np.sqrt(source_counts[s]) * 2.5 for s in top_sources],
        color=list(range(len(top_sources))),
        colorscale="Viridis",
        showscale=False,
        opacity=0.8,
    ),
    hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
)

# ── Assemble full HTML ──
heatmap_fig = go.Figure(heatmap)
heatmap_fig.update_layout(
    title="Franchise Personality Heatmap (top 25 franchises × 12 key traits)",
    xaxis_title="Trait",
    yaxis_title="Franchise",
    height=700,
    margin=dict(l=180),
    xaxis=dict(tickangle=-35),
)

scatter_fig = go.Figure(scatter)
scatter_fig.update_layout(
    title=f"Franchise Personality Space — PCA of 12 traits "
          f"(PC1={ev2[0]:.1f}%, PC2={ev2[1]:.1f}%)",
    xaxis_title=f"PC1 ({ev2[0]:.1f}%)",
    yaxis_title=f"PC2 ({ev2[1]:.1f}%)",
    height=520,
)

heatmap_div = heatmap_fig.to_html(include_plotlyjs=False, full_html=False, div_id="heatmap")
scatter_div = scatter_fig.to_html(include_plotlyjs=False, full_html=False, div_id="scatter")

# Radar dropdown chart will be built in JS
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Franchise Personality Profiles</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 20px 30px; background: #f8f8f8; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 30px; }}
  section {{ background: white; border-radius: 10px; padding: 20px 24px;
             margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; font-size: 18px; }}
  .radar-controls {{ display: flex; gap: 20px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }}
  select {{ padding: 7px 12px; font-size: 15px; border: 1px solid #bbb; border-radius: 4px; min-width: 200px; }}
  label {{ font-weight: 600; font-size: 14px; }}
  .explainer {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
               border-top: 1px solid #eee; margin-top: 24px; padding-top: 20px; }}
  .explainer-label {{ grid-column: 1/-1; font-size: 11px; font-weight: 700;
                      text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin-bottom: 2px; }}
  .exp-block h3 {{ font-size: 13px; font-weight: 700; color: #c87941; margin: 0 0 6px; }}
  .exp-block p {{ font-size: 13px; color: #555; line-height: 1.7; margin: 0; }}
</style>
</head>
<body>
<h1>Franchise Personality Profiles</h1>
<p class="subtitle">{len(scores)} characters from {len(top_sources)} franchises — scored on 12 personality dimensions</p>

<section>
  <h2>Radar: Compare Franchise Personality Profiles</h2>
  <div class="radar-controls">
    <div>
      <label>Franchise A:</label><br>
      <select id="selA"></select>
    </div>
    <div>
      <label>Franchise B (optional):</label><br>
      <select id="selB"><option value="">— none —</option></select>
    </div>
  </div>
  <div id="radar"></div>
  <div class="explainer">
    <div class="explainer-label">How to read this chart</div>
    <div class="exp-block"><h3>How it was made</h3><p>Each franchise's characters were averaged across 12 carefully chosen personality traits (out of 500 total). Those 12 — like Heroic, Kind, Wise, and Bold — were picked because they're the most interpretable and meaningful for comparing fictional universes.</p></div>
    <div class="exp-block"><h3>What you're looking at</h3><p>The radar chart is a spider-web shape. Each spoke represents one trait, and the distance from the center shows the average score for that franchise (0 = low, 100 = high). A wider shape means the franchise's characters are more extreme overall. Use the dropdowns to compare two franchises side-by-side.</p></div>
    <div class="exp-block"><h3>How to interpret it</h3><p>If Franchise A's shape bulges further than Franchise B's on the "Heroic" spoke, its characters are more heroic on average. Overlapping regions mean the two franchises are similar on those traits. A large gap on "Kind" or "Villainous" reveals a meaningful personality difference between the two shows.</p></div>
  </div>
</section>

<section>
  <h2>Heatmap: All Franchises at a Glance</h2>
  {heatmap_div}
  <div class="explainer">
    <div class="explainer-label">How to read this chart</div>
    <div class="exp-block"><h3>How it was made</h3><p>Every franchise's characters were averaged on 12 personality traits. The result is a single row per franchise showing its "average character." Green = high score on that trait, red = low score, yellow = near the midpoint (50/100).</p></div>
    <div class="exp-block"><h3>What you're looking at</h3><p>Rows are franchises, columns are traits. Each cell is one franchise's average score for one trait on a 0–100 scale. Hover over any cell to see the exact number. The color scale on the right shows what the colors mean.</p></div>
    <div class="exp-block"><h3>How to interpret it</h3><p>Scan down a column to find which franchise scores highest or lowest on a given trait — e.g. which show has the most villainous characters. Scan across a row to see the full personality fingerprint of a franchise. Lots of green = generally virtuous cast; lots of red = darker, morally complex show.</p></div>
  </div>
</section>

<section>
  <h2>Franchise Personality Space (PCA of 12 traits)</h2>
  <p style="color:#666;font-size:13px">Each dot = one franchise's average personality. Dot size = number of characters. Nearby = similar personality profile.</p>
  {scatter_div}
  <div class="explainer">
    <div class="explainer-label">How to read this chart</div>
    <div class="exp-block"><h3>How it was made</h3><p>PCA (Principal Component Analysis) took the 12-trait average profile of each franchise and compressed it into 2 numbers capturing the most variation. PC1 explains {ev2[0]:.1f}% of the differences between franchises; PC2 explains {ev2[1]:.1f}%. Together they capture most of what makes each franchise's cast unique.</p></div>
    <div class="exp-block"><h3>What you're looking at</h3><p>Each dot is one franchise, placed purely by personality data — no manual arrangement. Dot size reflects the number of characters from that franchise. The axes don't have simple labels because they blend all 12 traits together into composite directions.</p></div>
    <div class="exp-block"><h3>How to interpret it</h3><p>Franchises close together have similar average personality profiles — their casts "feel" alike in personality terms. Franchises far apart are most different. An outlier sitting alone has a uniquely distinct character mix compared to all other shows in the dataset.</p></div>
  </div>
</section>

<script>
const radarData = {json.dumps(radar_data)};
const traitLabels = {json.dumps(trait_labels)};
const franchiseNames = {json.dumps({s: nice_name(s) for s in top_sources})};
const sourceCounts = {json.dumps({s: int(source_counts[s]) for s in top_sources})};

const selA = document.getElementById('selA');
const selB = document.getElementById('selB');

{json.dumps(top_sources)}.forEach((src, i) => {{
  const nm = franchiseNames[src] || src;
  const cnt = sourceCounts[src] || 0;
  const opt = new Option(`${{nm}} (${{cnt}} chars)`, src);
  selA.add(opt.cloneNode(true));
  selB.add(opt);
}});
selA.value = {json.dumps(top_sources[0])};

function drawRadar() {{
  const a = selA.value;
  const b = selB.value;
  const traces = [];

  if (a && radarData[a]) {{
    const vals = radarData[a];
    traces.push({{
      type: 'scatterpolar',
      r: [...vals, vals[0]],
      theta: [...traitLabels, traitLabels[0]],
      fill: 'toself',
      name: franchiseNames[a] || a,
      opacity: 0.7,
    }});
  }}
  if (b && radarData[b]) {{
    const vals = radarData[b];
    traces.push({{
      type: 'scatterpolar',
      r: [...vals, vals[0]],
      theta: [...traitLabels, traitLabels[0]],
      fill: 'toself',
      name: franchiseNames[b] || b,
      opacity: 0.7,
    }});
  }}

  Plotly.react('radar', traces, {{
    polar: {{ radialaxis: {{ visible: true, range: [0, 100] }} }},
    showlegend: true,
    height: 480,
    margin: {{ t: 20, b: 20 }},
  }});
}}

selA.addEventListener('change', drawRadar);
selB.addEventListener('change', drawRadar);
drawRadar();
</script>
</body>
</html>"""

with open("franchise_profiles.html", "w") as f:
    f.write(html)

print("wrote franchise_profiles.html")
print(f"  {len(top_sources)} franchises, 12 traits")
print(f"\nTop 5 most heroic franchises:")
print(means_df["Heroic"].sort_values(ascending=False).head(5))
print(f"\nTop 5 most villainous franchises:")
print(means_df["Heroic"].sort_values().head(5))
