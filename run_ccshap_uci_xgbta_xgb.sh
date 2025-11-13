#!/bin/bash


mkdir ./outuci_xgbta
while read p;
do
  echo Processing $p;
  timeout 1d python ./src/expe_uci_classification_xgbta.py --id=$p --modelpath xgb_model --out outuci_xgbta --classifier=xgb;
done <datasets_xgb_all.txt

