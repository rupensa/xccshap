#!/bin/bash

mkdir ./outuci_sample_xgb

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_sample.py --id=$p --modelpath xgb_model --out outuci_sample_xgb --classifier=xgb;
done <datasets_sample.txt
