#!/bin/bash


mkdir ./outuci_xccshap_rf

while read p;
do
  echo Processing $p;
  timeout 1d python ./src/expe_uci_classification_xccshap.py --id=$p --modelpath rf_model --out outuci_xccshap_rf --classifier=rf;
done <datasets_xgb_all.txt

