import re
from pathlib import Path
import pandas as pd

DATA_ROOT = Path.home() / "Desktop" / "openpsychometrics"
SCORES_TSV = (
    DATA_ROOT
    / "SWCPQ-Features-Aggregated-Dataset-January2025"
    / "SWCPQ-Features-Aggregated-Dataset-January2025"
    / "data files"
    / "characters-aggregated-scores.csv"
)
KEY_JS = (
    DATA_ROOT
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "SWCPQ-Features-Survey-Dataset-November2023"
    / "resources"
    / "key.js"
)


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_TSV, sep="\t", index_col=0)
    df.index.name = "char_id"
    return df


def load_bap_labels() -> dict[str, tuple[str, str]]:
    text = KEY_JS.read_text()
    block = re.search(r"var measures\s*=\s*\{(.*?)\n\};", text, re.DOTALL).group(1)
    out = {}
    for m in re.finditer(r'(\d+)\s*:\s*\["([^"]+)","([^"]+)"\]', block):
        out[f"BAP{m.group(1)}"] = (m.group(2), m.group(3))
    return out


def load_character_names() -> pd.DataFrame:
    text = KEY_JS.read_text()
    rows = []
    for m in re.finditer(r'(\d+)\s*:\s*\["([^"]+)","(\w+)/(\d+)\.jpg"\]', text):
        source, num = m.group(3), m.group(4)
        rows.append({"char_id": f"{source}/{num}", "name": m.group(2), "source": source})
    return pd.DataFrame(rows).drop_duplicates("char_id").set_index("char_id")


if __name__ == "__main__":
    scores = load_scores()
    labels = load_bap_labels()
    names = load_character_names()
    print(f"scores: {scores.shape[0]} characters x {scores.shape[1]} traits")
    print(f"BAP labels: {len(labels)} pairs (e.g. BAP1 = {labels['BAP1']})")
    print(f"named characters: {len(names)}")
    print(f"Hermione row: {names.loc['HP/3', 'name']}")
