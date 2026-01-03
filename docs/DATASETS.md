# Datasets

## MedTech (PhysioNet AFDB)
- Source: PhysioNet AF Database (AFIB rhythm annotations)
- Access via WFDB PN (no credentials), record IDs like 04043, 07162
- Use: interval-based labeling on aux_note; not a certified medical device

## Industrial IoT (IMS Bearings)
- Source: NASA PCoE IMS Bearings (ASCII snapshots)
- Structure varies per archive; RMS series built per file; crest/kurtosis computed per segment
- Use: proxy metrics for early warning; RUL needs failure labels

## Cybersecurity (CIC IDS)
- Source: CICIDS2017 (pcap_ISCX.csv mirrors) for benign/DDoS flows
- Aggregation: second/minute; labels derived per window; balanced KPI requires BENIGN segments

## FinTech
- Source: yfinance (BTC-USD), public market data
- Use: structural regime detection; not financial advice

## Disclaimers
- Respect dataset licenses and terms
- No PII intended; use sanitized public datasets
- MedTech analytics are for decision support only; not for clinical diagnosis
