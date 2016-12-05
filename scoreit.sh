#!/bin/bash

END=9
for ((i=0;i<=END;i++)); do
    input="data/aueb_results/folds/aueb_results_fold$i"
    gold="scorers/gold_aueb/gold_fold_aueb$i"
    scorers/scoreA.pl $gold $input
done

input="data/aueb_results/folds/aueb_results_cross_fold_all"
gold="scorers/gold_aueb/gold_fold_cross_all"
scorers/scoreA.pl $gold $input



