# XCCSHAP
Source code for the paper "Explaining Random Forest and XGBoost with Shallow Decision Trees by Co-clustering Feature Importance", submitted to the Machine Learning journal.

## How to reproduce the experiments:

Fine-tune XGBoost and RF:

```
sh ./run_uci_grid_search.sh
```

Run the experiments for XCCSHAP:

```
sh ./run_ccshap_uci_xccshap.sh
```

Run the experiments for XGBTA:

```
sh ./run_ccshap_uci_xgbta.sh
```

Run the experiments for FBT:

```
sh ./run_ccshap_uci_fbtrf.sh
```

Run the experiments for sampling:

```
sh ./run_ccshap_uci_sample.sh
```


The results are in the output directories, one file per dataset.