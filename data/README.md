# Data

The small per-modality CSVs are shipped with the repo. The full synaptic edge list
(`connections_princeton.csv`, columns `pre_root_id, post_root_id, syn_count`) is too large to commit; 

```bash
python scripts/download_connectome.py
```

which downloads the public FlyWire FAFB v783 release and writes it to the path expected by `flynn.config.EDGE_PATH`.
