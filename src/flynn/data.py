from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests



from src.flynn.config import EDGE_PATH

FLYWIRE_CONNECTIONS_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz"


def ensure_edge_list(edge_list_path=None, url: str = FLYWIRE_CONNECTIONS_URL, force: bool = False) -> Path:
    edge_list_path = Path(edge_list_path or EDGE_PATH)
    if edge_list_path.exists() and not force:
        print(f"[data] {edge_list_path} already present ({edge_list_path.stat().st_size / 1e6:.1f} MB)")
        return edge_list_path

    edge_list_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = edge_list_path.with_suffix(".csv.gz")
    print(f"[data] downloading {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    df = pd.read_csv(gz_path)
    df.columns = [c.lower() for c in df.columns]
    df = df[["pre_root_id", "post_root_id", "syn_count"]]
    df.to_csv(edge_list_path, index=False)
    gz_path.unlink()
    print(f"[data] wrote {edge_list_path} ({edge_list_path.stat().st_size / 1e6:.1f} MB)")
    return edge_list_path
