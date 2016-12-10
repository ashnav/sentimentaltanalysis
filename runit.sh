#!/bin/bash
#Please see the run_classifier.py, train.py, and cross_validate.py files for overall code authorship
#author: Stephanie Durand

#specify the number of dimensions to use for the GloVe word embeddings vectors
d=100

#specify the number of folds to use for cross-validation
x=10

#specify all of the files to use
data_file=data/semeval_data.tsv
out_file=predictions.out

#run cross-fold validation
python code/cross_validate.py --d $d --x $x $out_file $data_file

#score the results
scorers/scoreA.pl $data_file $out_file

