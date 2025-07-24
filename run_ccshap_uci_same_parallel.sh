#!/bin/bash
set -o errexit

./run_ccshap_uci_same_single.sh datasets_xgb_all1.txt
pid1=$!
./run_ccshap_uci_same_single.sh datasets_xgb_all2.txt
pid2=$!
./run_ccshap_uci_same_single.sh datasets_xgb_all3.txt
pid3=$!
./run_ccshap_uci_same_single.sh datasets_xgb_all4.txt
pid4=$!
./run_ccshap_uci_same_single.sh datasets_xgb_all5.txt
pid5=$!

wait $pid1 && echo "pid1 exited normally" || echo "pid1 exited abnormally with status $?"
wait $pid2 && echo "pid2 exited normally" || echo "pid2 exited abnormally with status $?"
wait $pid3 && echo "pid3 exited normally" || echo "pid3 exited abnormally with status $?"
wait $pid4 && echo "pid2 exited normally" || echo "pid4 exited abnormally with status $?"
wait $pid5 && echo "pid3 exited normally" || echo "pid5 exited abnormally with status $?"
