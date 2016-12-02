#!/bin/bash
#Please see the run_classifier.py or train.py files for overall code authorship
#author: Stephanie Durand



#specify all of the files to use
train_file_2013=data/train/clean.tsv
dev_file_2013=data/dev/clean.tsv
train_file_2016=data/2016/clean_train.tsv
dev_file_2016=data/2016/clean_dev.tsv
test_file=data/2016/clean_devtest.tsv
out_file=predictions.out

all_tweets=data/semeval_data.tsv

#run the python
python code/weka/weka_data_builder.py $all_tweets

