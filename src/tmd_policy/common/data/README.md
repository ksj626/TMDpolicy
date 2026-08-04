# Data

`records.py` defines schema-v3 `ResearchRecord` and strict image validation; `store.py` writes append-only NPZ payloads with semantic and byte SHA-256; `loader.py` exposes a real split-filtered multi-record DataLoader; `splits.py` makes task-stratified episode-disjoint splits. Arrays retain producer dtype/normalization and object arrays are forbidden.

