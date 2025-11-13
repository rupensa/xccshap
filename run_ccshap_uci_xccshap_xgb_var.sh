#!/bin/bash


mkdir ./outuci_xccshap_xgb_var

while read p;
do
  echo Processing $p;
  timeout 1d python ./src/expe_uci_classification_xccshap_var.py --id=$p --modelpath xgb_model_var --out outuci_xccshap_xgb_var --classifier=xgb;
done <datasets_xgb_var.txt

