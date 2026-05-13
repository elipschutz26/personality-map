"""
explore_ranking_dataset.py
Explore the SWCPQ Ranking dataset — pairwise character similarity judgments
from real survey respondents. Finds which characters people perceive as most
similar (across and within franchises), and checks how well the BAP personality
scores predict human-perceived similarity.
"""
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from data_loader import load_scores, load_character_names

RANKING_CSV = (
    Path.home()
    / "Desktop/openpsychometrics"
    / "SWCPQ-Ranking-Survey-Dataset-December2022"
    / "SWCPQ-Ranking-Survey-Dataset-December2022"
    / "data files"
    / "SWCPQ-Ranking-Survey-Dataset.csv"
)
PICS_ROOT = (
    Path.home()
    / "Desktop/openpsychometrics"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "resources/pics"
)

scores = load_scores()
names = load_character_names()
id_to_name = {cid: (names.loc[cid, "name"] if cid in names.index else cid) for cid in scores.index}

def get_img(cid):
    parts = cid.split("/")
    if len(parts) == 2:
        f = PICS_ROOT / parts[0] / f"{parts[1]}.jpg"
        if f.exists():
            return f.as_uri()
    return None

print("Loading ranking data...")
df = pd.read_csv(RANKING_CSV, sep="\t", low_memory=False)
print(f"  {len(df)} survey responses loaded")

# Each row has a 'survey_ratings' column: JSON list of
# [charA, charB, bap_item_num, rating_1to6, time_ms]
# rating: 1=Extremely similar, 6=Extremely different → convert to similarity 0-100
# similarity = (6 - rating) / 5 * 100  → 100=Extremely Similar, 0=Extremely Different
pair_scores = defaultdict(list)
n_parsed = 0
for _, row in df.iterrows():
    raw = row.get("survey_ratings", "")
    if not isinstance(raw, str) or not raw.startswith("["):
        continue
    try:
        ratings = json.loads(raw)
    except Exception:
        try:
            ratings = ast.literal_eval(raw)
        except Exception:
            continue
    for entry in ratings:
        if len(entry) < 4:
            continue
        a, b, _bap_item, rating = entry[0], entry[1], entry[2], entry[3]
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 6:
            continue
        similarity = (6 - rating) / 5 * 100  # 100 = extremely similar
        key = tuple(sorted([str(a), str(b)]))
        pair_scores[key].append(similarity)
    n_parsed += 1

print(f"  {n_parsed} responses parsed, {len(pair_scores)} unique pairs")

# Build aggregated pair similarity (mean score, min 5 ratings for reliability)
MIN_RATINGS = 20
pair_agg = {
    k: {"mean": np.mean(v), "count": len(v), "std": np.std(v)}
    for k, v in pair_scores.items()
    if len(v) >= MIN_RATINGS
}
print(f"  {len(pair_agg)} pairs with >= {MIN_RATINGS} ratings")

# Sort by mean similarity score (highest = people thought most alike)
sorted_pairs = sorted(pair_agg.items(), key=lambda x: -x[1]["mean"])

# Only keep pairs where BOTH characters are in our scores dataset
known_ids = set(scores.index)
valid_pairs = [(k, v) for k, v in sorted_pairs if k[0] in known_ids and k[1] in known_ids]
print(f"  {len(valid_pairs)} valid pairs (both chars in scores dataset)")

# ── Panel 1: Top 30 most similar pairs ──
top30 = valid_pairs[:30]
pair_labels = []
pair_means = []
pair_sources = []
for (a, b), info in top30:
    src_a = a.split("/")[0]
    src_b = b.split("/")[0]
    cross = "cross" if src_a != src_b else src_a
    nm_a = id_to_name.get(a, a)
    nm_b = id_to_name.get(b, b)
    pair_labels.append(f"{nm_a} & {nm_b}")
    pair_means.append(round(info["mean"], 1))
    pair_sources.append(cross)

bar1 = go.Bar(
    y=pair_labels[::-1],
    x=pair_means[::-1],
    orientation="h",
    marker=dict(
        color=["#e74c3c" if s == "cross" else "#3498db" for s in pair_sources[::-1]],
        opacity=0.85,
    ),
    text=[f"{x:.1f}" for x in pair_means[::-1]],
    textposition="outside",
    hovertemplate="%{y}<br>Avg similarity: %{x:.1f}<extra></extra>",
)
fig1 = go.Figure(bar1)
fig1.update_layout(
    title="Top 30 Most Similar Character Pairs (human-rated)",
    xaxis_title="Average Similarity Score (0–100)",
    height=700,
    margin=dict(l=300),
    plot_bgcolor="#fafafa",
    annotations=[
        dict(x=0.99, y=0.02, xref="paper", yref="paper",
             text="<span style='color:#3498db'>■</span> Same franchise &nbsp; "
                  "<span style='color:#e74c3c'>■</span> Cross-franchise",
             showarrow=False, font=dict(size=12), xanchor="right"),
    ],
)

# ── Panel 2: BAP distance vs human similarity (correlation check) ──
bap_dist_matrix = pairwise_distances(scores.values)
char_ids_list = scores.index.tolist()
char_id_to_idx = {cid: i for i, cid in enumerate(char_ids_list)}

human_sim = []
bap_dist = []
for (a, b), info in valid_pairs[:2000]:  # sample for performance
    if a not in char_id_to_idx or b not in char_id_to_idx:
        continue
    ia, ib = char_id_to_idx[a], char_id_to_idx[b]
    human_sim.append(info["mean"])
    bap_dist.append(bap_dist_matrix[ia, ib])

corr = np.corrcoef(human_sim, bap_dist)[0, 1]
print(f"  Correlation (human similarity vs BAP distance): {corr:.3f}")

scatter2 = go.Scatter(
    x=bap_dist,
    y=human_sim,
    mode="markers",
    marker=dict(size=4, color="#3498db", opacity=0.3),
    hovertemplate="BAP distance: %{x:.1f}<br>Human similarity: %{y:.1f}<extra></extra>",
)
fig2 = go.Figure(scatter2)
fig2.update_layout(
    title=f"Do Personality Scores Predict Human Perception? (r = {corr:.2f})",
    xaxis_title="BAP Personality Distance (lower = more similar scores)",
    yaxis_title="Human-Rated Similarity (higher = more similar)",
    height=450,
    plot_bgcolor="#fafafa",
    annotations=[dict(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"Pearson r = {corr:.2f}",
        showarrow=False, font=dict(size=14, color="#e74c3c"), xanchor="right",
    )],
)

# ── Panel 3: Most surprising cross-franchise similarities ──
cross_pairs = [(k, v) for k, v in valid_pairs if k[0].split("/")[0] != k[1].split("/")[0]]
top_cross = cross_pairs[:20]

cross_labels = []
cross_vals = []
cross_hover = []
for (a, b), info in top_cross:
    nm_a = id_to_name.get(a, a)
    nm_b = id_to_name.get(b, b)
    src_a = a.split("/")[0]
    src_b = b.split("/")[0]
    cross_labels.append(f"{nm_a} ({src_a}) &\n{nm_b} ({src_b})")
    cross_vals.append(round(info["mean"], 1))
    cross_hover.append(f"{nm_a} ({src_a}) & {nm_b} ({src_b})<br>"
                       f"Rated {info['mean']:.1f}/100 similar<br>"
                       f"by {info['count']} respondents")

bar3 = go.Bar(
    y=cross_labels[::-1],
    x=cross_vals[::-1],
    orientation="h",
    marker=dict(color="#e74c3c", opacity=0.85),
    text=[f"{x:.1f}" for x in cross_vals[::-1]],
    textposition="outside",
    hovertext=cross_hover[::-1],
    hoverinfo="text",
)
fig3 = go.Figure(bar3)
fig3.update_layout(
    title="Top 20 Most Similar CROSS-Franchise Pairs",
    xaxis_title="Average Similarity Score (0–100)",
    height=600,
    margin=dict(l=320),
    plot_bgcolor="#fafafa",
)

bar1_div = fig1.to_html(include_plotlyjs=False, full_html=False, div_id="bar1")
scatter2_div = fig2.to_html(include_plotlyjs=False, full_html=False, div_id="scatter2")
bar3_div = fig3.to_html(include_plotlyjs=False, full_html=False, div_id="bar3")

# Portrait cards for top 10 pairs
def portrait_card(cid):
    nm = id_to_name.get(cid, cid)
    src = cid.split("/")[0]
    img = get_img(cid)
    img_el = (f'<img src="{img}" alt="{nm}" style="width:70px;height:70px;object-fit:cover;border-radius:8px;">'
              if img else '<div style="width:70px;height:70px;background:#ddd;border-radius:8px;"></div>')
    return f'''<div style="text-align:center;flex-shrink:0">
      {img_el}
      <div style="font-size:11px;margin-top:4px;max-width:70px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{nm}</div>
      <div style="font-size:10px;color:#888">{src}</div>
    </div>'''

pair_cards_html = ""
for (a, b), info in valid_pairs[:10]:
    nm_a = id_to_name.get(a, a)
    nm_b = id_to_name.get(b, b)
    pair_cards_html += f'''
    <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;background:#f9f9f9;border-radius:10px;margin-bottom:10px;border:1px solid #eee">
      {portrait_card(a)}
      <div style="text-align:center;flex:1">
        <div style="font-size:22px">≈</div>
        <div style="font-size:13px;font-weight:600">{info['mean']:.0f}% similar</div>
        <div style="font-size:11px;color:#888">{info['count']} ratings</div>
      </div>
      {portrait_card(b)}
    </div>'''

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ranking Dataset Insights</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; padding: 24px 36px; background: #f8f8f8; color: #222; max-width: 1100px; margin: 0 auto; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 30px; font-size: 14px; }}
  section {{ background: white; border-radius: 10px; padding: 20px 24px;
             margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  h2 {{ margin-top: 0; font-size: 18px; }}
  .stat-row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
  .stat-box {{ flex: 1; padding: 16px; background: #f5f5f5; border-radius: 8px; text-align: center; }}
  .stat-num {{ font-size: 28px; font-weight: 700; color: #2c3e50; }}
  .stat-label {{ font-size: 13px; color: #666; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Human-Rated Character Similarity</h1>
<p class="subtitle">Analysis of the SWCPQ Ranking Survey — {n_parsed:,} respondents rated {len(pair_agg):,} character pairs</p>

<div class="stat-row">
  <div class="stat-box">
    <div class="stat-num">{n_parsed:,}</div>
    <div class="stat-label">Survey Respondents</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{len(pair_scores):,}</div>
    <div class="stat-label">Unique Pairs Rated</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{len(valid_pairs):,}</div>
    <div class="stat-num" style="font-size:14px;color:#888">≥{MIN_RATINGS} ratings</div>
    <div class="stat-label">Reliable Pairs</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{corr:.2f}</div>
    <div class="stat-label">Correlation: BAP scores → human perception</div>
  </div>
</div>

<section>
  <h2>Most Similar Character Pairs (Portrait Gallery)</h2>
  <p style="color:#666;font-size:13px">Top 10 pairs rated most similar by survey respondents</p>
  {pair_cards_html}
</section>

<section>
  <h2>Top 30 Most Similar Pairs</h2>
  {bar1_div}
</section>

<section>
  <h2>Do Personality Scores Predict Human Perception?</h2>
  <p style="color:#666;font-size:13px">Each dot = one character pair. We'd expect: closer BAP scores → higher human similarity rating.</p>
  {scatter2_div}
</section>

<section>
  <h2>Most Surprising Cross-Franchise Similarities</h2>
  <p style="color:#666;font-size:13px">Characters from <em>different</em> shows/movies that people rated as highly similar</p>
  {bar3_div}
</section>
</body>
</html>"""

with open("ranking_insights.html", "w") as f:
    f.write(html)

print("\nwrote ranking_insights.html")
print(f"\nTop 5 most similar pairs:")
for (a, b), info in valid_pairs[:5]:
    print(f"  {id_to_name.get(a,a)} & {id_to_name.get(b,b)}: {info['mean']:.1f} (n={info['count']})")
print(f"\nBAP ↔ human similarity correlation: {corr:.3f}")
