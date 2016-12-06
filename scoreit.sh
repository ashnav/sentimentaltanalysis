#!/bin/bash

END=9
for ((i=0;i<=END;i++)); do
    input="code/word_vector_classifier/wv_fold$i"
    gold="scorers/gold_fold_weka$i"
    scorers/scoreA.pl $gold $input
done

#input="data/pruned_cross_fold/pruned_cross_fold_all"
#gold="scorers/gold_pipeline/gold_fold_cross_all"
#scorers/scoreA.pl $gold $input



