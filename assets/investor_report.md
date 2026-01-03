## Baseline Comparison
```json
[
  {
    "case": "FinTech",
    "metrics": {
      "rule_based": {
        "latency_ms": 25.0,
        "false_positives": 329,
        "false_negatives": 0,
        "stability": 0.9971428571428571,
        "compute_cost_ms_per_1M_events": 73821.71428097147
      },
      "autoencoder": {
        "latency_ms": 25.0,
        "false_positives": 329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 243569.71428330458
      },
      "lstm": {
        "latency_ms": 25.0,
        "false_positives": 329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 862966.2856193525
      },
      "tcn": {
        "latency_ms": 25.0,
        "false_positives": 329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 1009074.8573254262
      },
      "transformer": {
        "latency_ms": 25.0,
        "false_positives": 329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 1133281.71428293
      }
    }
  },
  {
    "case": "MedTech",
    "metrics": {
      "rule_based": {
        "latency_ms": 10.0,
        "false_positives": 738,
        "false_negatives": 0,
        "stability": 0.9905263157894737,
        "compute_cost_ms_per_1M_events": 92062.31585458705
      },
      "autoencoder": {
        "latency_ms": 850.0,
        "false_positives": 0,
        "false_negatives": 147,
        "stability": 0.9852631578947368,
        "compute_cost_ms_per_1M_events": 218319.89484397988
      },
      "lstm": {
        "latency_ms": 10.0,
        "false_positives": 2,
        "false_negatives": 131,
        "stability": 0.9947368421052631,
        "compute_cost_ms_per_1M_events": 834747.9999065399
      },
      "tcn": {
        "latency_ms": 10.0,
        "false_positives": 9,
        "false_negatives": 133,
        "stability": 0.9936842105263158,
        "compute_cost_ms_per_1M_events": 1019495.0526286113
      },
      "transformer": {
        "latency_ms": 10.0,
        "false_positives": 266,
        "false_negatives": 124,
        "stability": 0.9463157894736842,
        "compute_cost_ms_per_1M_events": 1070986.6315224452
      }
    }
  },
  {
    "case": "Industrial_IoT",
    "metrics": {
      "rule_based": {
        "latency_ms": 10.0,
        "false_positives": 551,
        "false_negatives": 0,
        "stability": 0.9863157894736843,
        "compute_cost_ms_per_1M_events": 89938.52638492458
      },
      "autoencoder": {
        "latency_ms": 260.0,
        "false_positives": 0,
        "false_negatives": 26,
        "stability": 0.9978947368421053,
        "compute_cost_ms_per_1M_events": 212369.15803073268
      },
      "lstm": {
        "latency_ms": 90.0,
        "false_positives": 1,
        "false_negatives": 179,
        "stability": 0.9957894736842106,
        "compute_cost_ms_per_1M_events": 843726.3159277408
      },
      "tcn": {
        "latency_ms": 40.0,
        "false_positives": 1,
        "false_negatives": 9,
        "stability": 0.9978947368421053,
        "compute_cost_ms_per_1M_events": 991781.3684105088
      },
      "transformer": {
        "latency_ms": 30.0,
        "false_positives": 247,
        "false_negatives": 71,
        "stability": 0.9621052631578947,
        "compute_cost_ms_per_1M_events": 1075632.5263431983
      }
    }
  },
  {
    "case": "IT_Infrastructure",
    "metrics": {
      "rule_based": {
        "latency_ms": 1000.0,
        "false_positives": 1309,
        "false_negatives": 0,
        "stability": 0.9654676258992806,
        "compute_cost_ms_per_1M_events": 87943.95671822398
      },
      "autoencoder": {
        "latency_ms": 1000.0,
        "false_positives": 1329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 193723.66904783592
      },
      "lstm": {
        "latency_ms": 1000.0,
        "false_positives": 1329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 819466.4747809335
      },
      "tcn": {
        "latency_ms": 1000.0,
        "false_positives": 1329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 1042598.6330152202
      },
      "transformer": {
        "latency_ms": 1000.0,
        "false_positives": 1329,
        "false_negatives": 0,
        "stability": 1.0,
        "compute_cost_ms_per_1M_events": 1074944.3884764751
      }
    }
  }
]
```

## Stress Summary
```json
[
  {
    "case": "FinTech",
    "stability": {
      "rule_white": 0.9742857142857143,
      "lstm_white": 1.0,
      "tcn_white": 1.0,
      "transformer_white": 1.0,
      "autoencoder_white": 1.0,
      "rule_pink": 1.0,
      "lstm_pink": 1.0,
      "tcn_pink": 1.0,
      "transformer_pink": 1.0,
      "autoencoder_pink": 1.0,
      "rule_burst": 1.0,
      "lstm_burst": 1.0,
      "tcn_burst": 1.0,
      "transformer_burst": 1.0,
      "autoencoder_burst": 1.0,
      "rule_missing_0.05": 0.9428571428571428,
      "lstm_missing_0.05": 1.0,
      "tcn_missing_0.05": 0.9085714285714286,
      "transformer_missing_0.05": 1.0,
      "autoencoder_missing_0.05": 1.0,
      "rule_missing_0.1": 0.9371428571428572,
      "lstm_missing_0.1": 1.0,
      "tcn_missing_0.1": 0.8942857142857142,
      "transformer_missing_0.1": 1.0,
      "autoencoder_missing_0.1": 1.0,
      "rule_missing_0.3": 0.26285714285714284,
      "lstm_missing_0.3": 1.0,
      "tcn_missing_0.3": 0.8314285714285714,
      "transformer_missing_0.3": 1.0,
      "autoencoder_missing_0.3": 1.0,
      "rule_shift_5": 0.9285714285714286,
      "lstm_shift_5": 1.0,
      "tcn_shift_5": 0.9885714285714285,
      "transformer_shift_5": 1.0,
      "autoencoder_shift_5": 1.0,
      "rule_shift_10": 0.9285714285714286,
      "lstm_shift_10": 1.0,
      "tcn_shift_10": 0.9742857142857143,
      "transformer_shift_10": 1.0,
      "autoencoder_shift_10": 1.0,
      "rule_drift_0.1": 0.9914285714285714,
      "lstm_drift_0.1": 1.0,
      "tcn_drift_0.1": 0.9771428571428571,
      "transformer_drift_0.1": 1.0,
      "autoencoder_drift_0.1": 1.0,
      "rule_drift_0.3": 0.9485714285714286,
      "lstm_drift_0.3": 1.0,
      "tcn_drift_0.3": 0.9457142857142857,
      "transformer_drift_0.3": 1.0,
      "autoencoder_drift_0.3": 1.0
    }
  },
  {
    "case": "MedTech",
    "stability": {
      "rule_white": 0.9726315789473684,
      "lstm_white": 0.9810526315789474,
      "tcn_white": 0.9715789473684211,
      "transformer_white": 1.0,
      "autoencoder_white": 0.9989473684210526,
      "rule_pink": 0.9747368421052631,
      "lstm_pink": 0.9831578947368421,
      "tcn_pink": 0.9863157894736843,
      "transformer_pink": 1.0,
      "autoencoder_pink": 0.9968421052631579,
      "rule_burst": 0.9968421052631579,
      "lstm_burst": 0.9736842105263158,
      "tcn_burst": 0.9673684210526315,
      "transformer_burst": 1.0,
      "autoencoder_burst": 0.9989473684210526,
      "rule_missing_0.05": 0.9494736842105264,
      "lstm_missing_0.05": 0.94,
      "tcn_missing_0.05": 0.9431578947368421,
      "transformer_missing_0.05": 1.0,
      "autoencoder_missing_0.05": 0.9831578947368421,
      "rule_missing_0.1": 0.9494736842105264,
      "lstm_missing_0.1": 0.9094736842105263,
      "tcn_missing_0.1": 0.9442105263157895,
      "transformer_missing_0.1": 1.0,
      "autoencoder_missing_0.1": 0.9736842105263158,
      "rule_missing_0.3": 0.9494736842105264,
      "lstm_missing_0.3": 0.8589473684210527,
      "tcn_missing_0.3": 0.9431578947368421,
      "transformer_missing_0.3": 1.0,
      "autoencoder_missing_0.3": 0.888421052631579,
      "rule_shift_5": 0.968421052631579,
      "lstm_shift_5": 0.911578947368421,
      "tcn_shift_5": 0.9094736842105263,
      "transformer_shift_5": 1.0,
      "autoencoder_shift_5": 0.9715789473684211,
      "rule_shift_10": 0.9494736842105264,
      "lstm_shift_10": 0.8263157894736842,
      "tcn_shift_10": 0.9421052631578948,
      "transformer_shift_10": 1.0,
      "autoencoder_shift_10": 0.9568421052631579,
      "rule_drift_0.1": 0.9989473684210526,
      "lstm_drift_0.1": 0.9705263157894737,
      "tcn_drift_0.1": 0.9757894736842105,
      "transformer_drift_0.1": 1.0,
      "autoencoder_drift_0.1": 0.9852631578947368,
      "rule_drift_0.3": 0.9978947368421053,
      "lstm_drift_0.3": 0.9536842105263158,
      "tcn_drift_0.3": 0.9547368421052631,
      "transformer_drift_0.3": 1.0,
      "autoencoder_drift_0.3": 0.8010526315789473
    }
  },
  {
    "case": "Industrial_IoT",
    "stability": {
      "rule_white": 0.9989473684210526,
      "lstm_white": 0.9789473684210527,
      "tcn_white": 0.9105263157894737,
      "transformer_white": 1.0,
      "autoencoder_white": 0.9978947368421053,
      "rule_pink": 0.9989473684210526,
      "lstm_pink": 0.9789473684210527,
      "tcn_pink": 0.8652631578947368,
      "transformer_pink": 1.0,
      "autoencoder_pink": 1.0,
      "rule_burst": 1.0,
      "lstm_burst": 0.9378947368421052,
      "tcn_burst": 0.9105263157894737,
      "transformer_burst": 1.0,
      "autoencoder_burst": 0.9968421052631579,
      "rule_missing_0.05": 0.9989473684210526,
      "lstm_missing_0.05": 0.871578947368421,
      "tcn_missing_0.05": 0.8547368421052631,
      "transformer_missing_0.05": 1.0,
      "autoencoder_missing_0.05": 0.991578947368421,
      "rule_missing_0.1": 0.9989473684210526,
      "lstm_missing_0.1": 0.8578947368421053,
      "tcn_missing_0.1": 0.848421052631579,
      "transformer_missing_0.1": 1.0,
      "autoencoder_missing_0.1": 0.9905263157894737,
      "rule_missing_0.3": 0.9989473684210526,
      "lstm_missing_0.3": 0.783157894736842,
      "tcn_missing_0.3": 0.8557894736842105,
      "transformer_missing_0.3": 1.0,
      "autoencoder_missing_0.3": 0.9842105263157894,
      "rule_shift_5": 0.9978947368421053,
      "lstm_shift_5": 0.43157894736842106,
      "tcn_shift_5": 0.8063157894736842,
      "transformer_shift_5": 1.0,
      "autoencoder_shift_5": 0.9905263157894737,
      "rule_shift_10": 0.9978947368421053,
      "lstm_shift_10": 0.9315789473684211,
      "tcn_shift_10": 0.8473684210526315,
      "transformer_shift_10": 1.0,
      "autoencoder_shift_10": 0.98,
      "rule_drift_0.1": 1.0,
      "lstm_drift_0.1": 0.9578947368421052,
      "tcn_drift_0.1": 0.9852631578947368,
      "transformer_drift_0.1": 1.0,
      "autoencoder_drift_0.1": 0.9936842105263158,
      "rule_drift_0.3": 0.9989473684210526,
      "lstm_drift_0.3": 0.92,
      "tcn_drift_0.3": 0.8610526315789474,
      "transformer_drift_0.3": 1.0,
      "autoencoder_drift_0.3": 0.8105263157894737
    }
  },
  {
    "case": "IT_Infrastructure",
    "stability": {
      "rule_white": 0.9978417266187051,
      "lstm_white": 1.0,
      "tcn_white": 1.0,
      "transformer_white": 1.0,
      "autoencoder_white": 1.0,
      "rule_pink": 1.0,
      "lstm_pink": 1.0,
      "tcn_pink": 1.0,
      "transformer_pink": 1.0,
      "autoencoder_pink": 1.0,
      "rule_burst": 1.0,
      "lstm_burst": 1.0,
      "tcn_burst": 1.0,
      "transformer_burst": 1.0,
      "autoencoder_burst": 1.0,
      "rule_missing_0.05": 0.9935251798561151,
      "lstm_missing_0.05": 1.0,
      "tcn_missing_0.05": 0.9964028776978417,
      "transformer_missing_0.05": 1.0,
      "autoencoder_missing_0.05": 1.0,
      "rule_missing_0.1": 0.9719424460431655,
      "lstm_missing_0.1": 1.0,
      "tcn_missing_0.1": 0.9928057553956835,
      "transformer_missing_0.1": 1.0,
      "autoencoder_missing_0.1": 1.0,
      "rule_missing_0.3": 0.1683453237410072,
      "lstm_missing_0.3": 1.0,
      "tcn_missing_0.3": 0.9870503597122302,
      "transformer_missing_0.3": 1.0,
      "autoencoder_missing_0.3": 1.0,
      "rule_shift_5": 0.9870503597122302,
      "lstm_shift_5": 1.0,
      "tcn_shift_5": 0.9928057553956835,
      "transformer_shift_5": 1.0,
      "autoencoder_shift_5": 1.0,
      "rule_shift_10": 0.9856115107913669,
      "lstm_shift_10": 1.0,
      "tcn_shift_10": 0.9856115107913669,
      "transformer_shift_10": 1.0,
      "autoencoder_shift_10": 1.0,
      "rule_drift_0.1": 0.9985611510791367,
      "lstm_drift_0.1": 1.0,
      "tcn_drift_0.1": 0.9841726618705036,
      "transformer_drift_0.1": 1.0,
      "autoencoder_drift_0.1": 1.0,
      "rule_drift_0.3": 0.9942446043165467,
      "lstm_drift_0.3": 1.0,
      "tcn_drift_0.3": 0.9532374100719424,
      "transformer_drift_0.3": 1.0,
      "autoencoder_drift_0.3": 1.0
    }
  }
]
```

## Explainability Summary
```json
[
  {
    "case": "FinTech",
    "inv_weight_avg": 0.5212010874718505,
    "inv_weight_max": 0.9787488970623798,
    "spectral_entropy_avg": 0.012481230120340716,
    "correlation_with_event": 0.2584614287042776
  },
  {
    "case": "MedTech",
    "inv_weight_avg": 0.5430048865749287,
    "inv_weight_max": 0.9931036244460979,
    "spectral_entropy_avg": 1.4656976028351811,
    "correlation_with_event": 0.4523362848878018
  },
  {
    "case": "Industrial_IoT",
    "inv_weight_avg": 0.7135844805298027,
    "inv_weight_max": 0.8977151888562028,
    "spectral_entropy_avg": 0.7421616244368072,
    "correlation_with_event": 0.6685353933919275
  },
  {
    "case": "IT_Infrastructure",
    "inv_weight_avg": 0.4802157572650278,
    "inv_weight_max": 1.0726742479445035,
    "spectral_entropy_avg": 0.046944859441491585,
    "correlation_with_event": 0.00024883919289831255
  }
]
```

## Latency Benchmark
```json
{
  "controller_ms": {
    "mean_ms": 6.472221998763936,
    "p95_ms": 8.204140071757138,
    "p99_ms": 9.144831073936075,
    "min_ms": 2.9396999161690474,
    "max_ms": 14.284900156781077
  },
  "model_ms": {
    "mean_ms": 10.956231006421149,
    "p95_ms": 13.293260091450062,
    "p99_ms": 14.838753810618055,
    "min_ms": 4.145000129938126,
    "max_ms": 22.23950019106269
  }
}
```

## Scalability Summary
```json
{
  "10k": {
    "windows": 10000,
    "mean_ms": 3.319813790381886,
    "p95_ms": 7.09901504451409,
    "throughput_events_per_sec": 300.747051466524,
    "total_time_s": 33.250533799873665
  },
  "100k": {
    "windows": 100000,
    "mean_ms": 3.368740231315605,
    "p95_ms": 8.044005010742692,
    "throughput_events_per_sec": 296.3735709410655,
    "total_time_s": 337.4120022999123
  },
  "1M_extrapolated_total_s": 3368.7402313156053
}
```

## FinTech Summary
```json
{
  "ticker": "BTC-USD",
  "regime": {
    "correct_regime_classification": 0.7461477151965994,
    "whipsaw_count": 22
  },
  "impact": {
    "max_drawdown_reduction": -9.992007221626409e-16,
    "trade_suppression_rate": 0.004117959617428267
  }
}
```

## MedTech Summary
```json
{
  "event": "real_ecg_stream",
  "metrics": {
    "inv_weight_avg": 0.47530955577270545,
    "inv_weight_max": 0.8094389941640311,
    "false_alarm_rate_per_hour": 3.9130434782608696
  }
}
```

## MedTech PhysioNet
```json
{
  "record": "07162",
  "fs": 250,
  "metrics": {
    "sensitivity": 0.9999999999978948,
    "specificity": 0.0,
    "precision": 0.9999999999978948,
    "recall": 0.9999999999978948,
    "false_alarm_rate_per_hour": 14.4,
    "tp": 4750,
    "fp": 0,
    "tn": 0,
    "fn": 0,
    "optimal_threshold": 0.2343968699468672,
    "f1": 0.9999999949978948
  }
}
```

## MedTech AFDB ROC
```json
[
  {
    "thr": 0.23439687490463257,
    "tpr": 0.9999999999978948,
    "fpr": 0.0
  },
  {
    "thr": 0.24790222942829132,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.2614075839519501,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.27491292357444763,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.2884182929992676,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.30192363262176514,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.3154289722442627,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.32893434166908264,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.3424397110939026,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.35594505071640015,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.3694503903388977,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.38295575976371765,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.3964610993862152,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.40996646881103516,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.4234718084335327,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.4369771480560303,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.4504825174808502,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.46398788690567017,
    "tpr": 0.8719999999981642,
    "fpr": 0.0
  },
  {
    "thr": 0.4774932265281677,
    "tpr": 0.773473684208898,
    "fpr": 0.0
  },
  {
    "thr": 0.4909985661506653,
    "tpr": 0.6694736842091169,
    "fpr": 0.0
  },
  {
    "thr": 0.5045039653778076,
    "tpr": 0.6694736842091169,
    "fpr": 0.0
  },
  {
    "thr": 0.5180093050003052,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.5315146446228027,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.5450199842453003,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.5585253238677979,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.5720306634902954,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.585536003112793,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.5990414023399353,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.6125467419624329,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.6260521411895752,
    "tpr": 0.6692631578933279,
    "fpr": 0.0
  },
  {
    "thr": 0.6395574808120728,
    "tpr": 0.44526315789379944,
    "fpr": 0.0
  },
  {
    "thr": 0.6530628204345703,
    "tpr": 0.27389473684152865,
    "fpr": 0.0
  },
  {
    "thr": 0.6665681600570679,
    "tpr": 0.2736842105257396,
    "fpr": 0.0
  },
  {
    "thr": 0.6800734996795654,
    "tpr": 0.2724210526310054,
    "fpr": 0.0
  },
  {
    "thr": 0.693578839302063,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7070841789245605,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7205895781517029,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7340949177742004,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.747600257396698,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7611056566238403,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7746109962463379,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.7881163358688354,
    "tpr": 0.2679999999994358,
    "fpr": 0.0
  },
  {
    "thr": 0.801621675491333,
    "tpr": 0.15515789473651545,
    "fpr": 0.0
  },
  {
    "thr": 0.8151270151138306,
    "tpr": 0.020842105263114016,
    "fpr": 0.0
  },
  {
    "thr": 0.8286324143409729,
    "tpr": 0.018736842105223712,
    "fpr": 0.0
  },
  {
    "thr": 0.8421377539634705,
    "tpr": 0.016842105263122437,
    "fpr": 0.0
  },
  {
    "thr": 0.855643093585968,
    "tpr": 0.016631578947333407,
    "fpr": 0.0
  },
  {
    "thr": 0.8691484332084656,
    "tpr": 0.016421052631544378,
    "fpr": 0.0
  },
  {
    "thr": 0.8826537728309631,
    "tpr": 0.016210526315755345,
    "fpr": 0.0
  },
  {
    "thr": 0.8961591720581055,
    "tpr": 0.00021052631578903047,
    "fpr": 0.0
  }
]
```

## MedTech AFDB Multi
```json
[]
```

## IoT Summary
```json
{
  "inv_weight_avg": 0.839953601360321,
  "inv_weight_max": 1.1503573656082153,
  "false_maintenance_triggers": 14463,
  "downtime_prevented_hours_proxy": 144.63
}
```

## IoT IMS KPI
```json
{
  "series_points": 19735,
  "lead_time_points_proxy": 17716.0,
  "false_maintenance_triggers_before_failure_proxy": 12904,
  "inv_weight_avg": 0.839953601360321,
  "inv_weight_max": 1.1503573656082153,
  "crest_factor_avg": 2.7405998923980195,
  "kurtosis_avg": 5.83348665289507
}
```

## Cybersecurity Summary
```json
{
  "false_block_rate": 0.0,
  "entropy_separation_score": -0.05064557459236674
}
```

## Cyber CIC KPI
```json
{
  "metrics": {
    "sensitivity": 0.9999999999401197,
    "specificity": 0.0,
    "precision": 0.9488636363097236,
    "recall": 0.9999999999401197,
    "tp": 167,
    "fp": 9,
    "tn": 0,
    "fn": 0,
    "false_positive_rate": 0.9999999988888888
  },
  "optimal_threshold": 0.10000000149011612,
  "selection_method": "youden_j",
  "aux_f1_metrics": {
    "sensitivity": 0.9999999999401197,
    "specificity": 0.0,
    "precision": 0.9488636363097236,
    "recall": 0.9999999999401197,
    "tp": 167,
    "fp": 9,
    "tn": 0,
    "fn": 0,
    "false_positive_rate": 0.9999999988888888
  },
  "aux_f1_threshold": 0.10000000149011612,
  "detection_latency_minutes": 1.0
}
```
