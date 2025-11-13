#!/bin/bash

mkdir ./xgb_model_var
mkdir ./rf_model_var

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_gridsearch_var.py --id=$p --out xgb_model_var --classifier=xgb;
  python ./src/expe_uci_gridsearch_var.py --id=$p --out rf_model_var --classifier=rf;
done <datasets_xgb_var.txt
