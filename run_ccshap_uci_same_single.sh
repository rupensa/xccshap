#!/bin/bash


while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_same.py --id=$p --modelpath rf_model --out outuci_same --classifier=rf;
done < $1

