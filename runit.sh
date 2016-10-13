#!/bin/bash

#specify the number of dimensions to use for the GloVe word embeddings vectors
d=100

#specify all of the files to use
train_file=data/train60
test_file=data/test40
out_file=predictions.out

#train the machine learning classifiers
python code/train.py $d $train_file

#classify the test data
python code/run_classifier.py $d $out_file $test_file


