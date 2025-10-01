#!/bin/bash


mkdir ./outuci_fbtrf
while read p;
do
  echo Processing $p;
  timeout 1d python ./src/expe_uci_classification_fbtrf.py --id=$p --modelpath rf_model --out outuci_fbtrf --classifier=rf;
done <datasets_xgb_all.txt

