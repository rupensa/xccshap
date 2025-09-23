#!/bin/bash


mkdir ./outuci_uniform_xgb

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_dummy.py --id=$p --modelpath xgb_model --out outuci_uniform_xgb --classifier=xgb;
done <datasets_xgb_all.txt

