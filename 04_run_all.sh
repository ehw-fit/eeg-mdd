#!/usr/bin/bash

resdir="res"

for class in "ridge" "kneighbors" "svm" "dt"; do
    echo $class 
    for i in $(seq 4) ; do
        r="$resdir/search_${class}_r${i}"
        python 03_ga_features_selector.py $r.pkl.gz \
            --p_size 15 --q_size 40 \
            --log $r.gz \
            --classifier $class \
            --generations 1000 2>&1 | tee $r.log &
        pids[${i}]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
done