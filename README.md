# XCCSHAP
Source code for the paper "Explaining Random Forest and XGBoost with Shallow Decision Trees by Co-clustering Feature Importance", submitted to the Machine Learning journal.

## How to reproduce the experiments (example):

Fine-tune XGBoost and RF:

```
sh ./run_uci_grid_search.sh
```

Run the experiments:

```
sh ./run_ccshap_uci_xccshap.sh
```


The results are in the output directories, one file per dataset.

The scripts are named according to the following template string:
```
run_ccshap_uci_<algorithm>_[classifier]_[experiment].sh
```
