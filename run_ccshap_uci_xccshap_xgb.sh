#!/bin/bash


mkdir ./outuci_xccshap_xgb

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_xccshap.py --id=$p --modelpath xgb_model --out outuci_xccshap_xgb --classifier=xgb;
done <datasets_xgb_all.txt

