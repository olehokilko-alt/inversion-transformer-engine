# Investor / Enterprise Package

## Overview
- Positioning: We don’t compete on prediction accuracy. We dominate structural change detection under noise.
- Reproducible scripts and generated artifacts for validation across 4 industries.

## Artifacts
- Whitepaper: [WHITEPAPER.md](WHITEPAPER.md)
- Case Studies: [CASE_STUDIES.md](CASE_STUDIES.md)
- Dataset References: [DATASETS.md](DATASETS.md)
- Baseline Comparison: [assets/baseline_comparison.json](../assets/baseline_comparison.json)
- Stress & Adversarial: [assets/stress_summary.json](../assets/stress_summary.json)
- Explainability: [assets/explainability_summary.json](../assets/explainability_summary.json)
- Latency Benchmark: [assets/latency_benchmark.json](../assets/latency_benchmark.json)
- Scalability: [assets/scalability_summary.json](../assets/scalability_summary.json)
- Latency Figure: [assets/latency_plot.png](../assets/latency_plot.png)
- Scalability Figure: [assets/scalability_plot.png](../assets/scalability_plot.png)
- FinTech: [assets/fintech_summary.json](../assets/fintech_summary.json)
- MedTech: [assets/medtech_summary.json](../assets/medtech_summary.json)
- MedTech (PhysioNet AFDB): [assets/medtech_physionet_summary.json](../assets/medtech_physionet_summary.json)
- MedTech AFDB ROC: [assets/medtech_afdb_roc.json](../assets/medtech_afdb_roc.json)
- MedTech AFDB Multi-Record: [assets/medtech_afdb_multi.json](../assets/medtech_afdb_multi.json)
- MedTech AFDB Balanced: [assets/medtech_afdb_balanced.json](../assets/medtech_afdb_balanced.json)
- Industrial IoT: [assets/iot_summary.json](../assets/iot_summary.json)
- Industrial IoT (IMS KPI): [assets/iot_ims_kpi.json](../assets/iot_ims_kpi.json)
- Cybersecurity: [assets/cybersec_summary.json](../assets/cybersec_summary.json)
- Cybersecurity (CIC KPI): [assets/cyber_cic_kpi.json](../assets/cyber_cic_kpi.json)
- Consolidated Report: [assets/investor_report.md](../assets/investor_report.md)

## Reproducible Commands
```bash
python run_all.py
python universal_test_suite.py
python benchmarks/baseline_comparison_suite.py
python benchmarks/stress_tests.py
python benchmarks/explainability.py
python benchmarks/latency_benchmark.py
python benchmarks/scalability_test.py
python benchmarks/fintech_tests.py
python benchmarks/medtech_tests.py
python benchmarks/iot_tests.py
python benchmarks/cybersec_tests.py
python benchmarks/kpi_table.py
python benchmarks/build_figures.py
python benchmarks/report_builder.py
python benchmarks/medtech_afdb_roc.py
python benchmarks/medtech_afdb_multi.py
python benchmarks/medtech_afdb_balanced.py
```

## KPI per Industry
- Finance: Reduce structural false signals vs. vanilla Transformer at equal latency.
- MedTech: Reduces false alarms at equal sensitivity on AFib‑like events.
- Industrial IoT: Detects faults earlier than vibration‑threshold systems.
- Cybersecurity: Detects DDoS without IP reputation/signatures with lower false positives.

## Notes
- Stress tests include noise injection, missing data, time shift and drift.
- Explainability provides Inversion Weight timelines and spectral entropy traces.
