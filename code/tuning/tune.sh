#!/bin/bash
conf_sp1="conf_sp1_new"
conf_sp2="conf_sp2_new"
conf_pipe="conf_pipe_new"
gold="../../data/2016/clean_devtest.tsv"

python tuning_script.py $gold $conf_sp1 $conf_sp2 $conf_pipe 
