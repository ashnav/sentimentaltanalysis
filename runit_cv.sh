#!/bin/bash
#Please see the run_classifier.py or train.py files for overall code authorship
#author: Stephanie Durand

#specify the number of "folds" for crossfold validation
x=10
#specify the number of dimensions to use for the GloVe word embeddings vectors
d=100


#specify all of the files to use
train_file_2013=data/train/clean.tsv
dev_file_2013=data/dev/clean.tsv
train_file_2016=data/2016/clean_train.tsv
dev_file_2016=data/2016/clean_dev.tsv
test_file=data/2016/clean_devtest.tsv
out_file=predictions.out

#perform 10-fold cross validation
python code/cross_validate.py $x $d $train_file_2013 $dev_file_2013 $train_file_2016 $dev_file_2016 $test_file



