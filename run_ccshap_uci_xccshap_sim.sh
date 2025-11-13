#!/bin/bash


while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_xccshap_sim.py --id=$p --modelpath xgb_model --out outuci_xccshap_xgb --classifier=xgb;
  python ./src/expe_uci_classification_xccshap_sim.py --id=$p --modelpath rf_model --out outuci_xccshap_rf --classifier=rf;
done <datasets_xgb_all.txt

