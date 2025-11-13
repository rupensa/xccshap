#!/bin/bash


mkdir ./outuci_xccshap_rf_var

while read p;
do
  echo Processing $p;
  timeout 1d python ./src/expe_uci_classification_xccshap_var.py --id=$p --modelpath rf_model_var --out outuci_xccshap_rf_var --classifier=rf;
done <datasets_xgb_var.txt

