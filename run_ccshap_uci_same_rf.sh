#!/bin/bash


mkdir ./outuci_same
while read p;
do
  echo Processing $p;
  timeout 3d python ./src/expe_uci_classification_same.py --id=$p --modelpath rf_model --out outuci_same --classifier=rf;
done <datasets_xgb_all.txt

