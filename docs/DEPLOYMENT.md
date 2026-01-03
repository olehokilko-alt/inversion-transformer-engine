# Deployment Guide

## Requirements
- Python 3.10+
- pip, virtualenv (optional)
- OS: Windows/Linux/macOS
- RAM: ≥8 GB (recommended 16 GB for large datasets)
- CPU: ≥4 cores (recommended 8+ for throughput scaling)

## Install
```bash
python -m pip install -r requirements.txt
```

## Run (Quick)
```bash
python data_fetch.py
python run_all.py
python benchmarks/kpi_table.py
python benchmarks/build_figures.py
python benchmarks/report_builder.py
```

## Per-Industry KPIs
```bash
# MedTech
python benchmarks/medtech_afdb_balanced.py
python benchmarks/medtech_afdb_roc.py
# IoT
python benchmarks/iot_ims_kpi.py
# Cyber
python benchmarks/cyber_cic_kpi.py
python benchmarks/cyber_cic_balanced_kpi.py
```

## Performance Tips
- Use smaller seq_len where acceptable
- Batch computations and avoid Python loops where possible
- Consider multiprocessing for window batches
- Optional: Numba for hot paths

## Notes
- Data scripts download public datasets; ensure network access
- No secrets in repo; configure credentials only for private datasets (not required here)
