#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
############## Semeval - Sentiment Analysis in Twitter  #################
#########################################################################

####
#### Authors: Pedro Paulo Balage Filho e Lucas Avanço
#### Modifications made by: Stephanie Durand
#### Version: 2.0
#### Date: 26/03/14
####

# Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

import codecs
import os

#### Class to provide a data reader for Semeval 2014 Task 9 format.
#### The reader was designed for subtask A (sentiment for the twitter message) only.
#### Information about Semeval format can be found at:
####    http://alt.qcri.org/semeval2014/task9/
####
class SemevalTwitter(object):

    # Constructor.
    def __init__(self,train_path,dev_path,test_path):

        # class variables
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.trainset = list()
        self.devset = list()
        self.testset = list()
        self.reader()
        
        
    #read semeval dataset format    
    def read_file(self, f):
      fp = codecs.open(f, 'r', encoding='utf8')
      lines = fp.readlines()
      tweets = []
      for line in lines:
          line = line.strip()
          tweet_line = line.split('\t')
          if len(tweet_line) != 3:
              print('Error to read TrainSet. Must have 3 args. Line: ', line)
          tweet = {}
          tweet['ID'] = tweet_line[0]
          sentiment = tweet_line[1]
          # classes objective and neutral merged as proposed in the task
          if sentiment in ['objective','objective-OR-neutral']:
              sentiment = 'neutral'
          tweet['SENTIMENT'] = sentiment
          tweet['MESSAGE'] = tweet_line[2].strip()
          if tweet['MESSAGE'] != 'Not Available':
              tweets.append(tweet)
      return tweets 
 
    # read semeval dataset format
    # modifies the train, dev and testset class variables
    def reader(self):

        # Read the trainset
        tweets = []
        if not os.path.exists(self.train_path):
            print('Trainset file not found. However, they are not necessary if provided the training model (file: model_pythonx.pkl)')
        else:
            tweets = self.read_file(self.train_path)
        self.trainset = tweets

        # Read the devset
        tweets = []
        if not os.path.exists(self.dev_path):
            print('Devset file not found. However, they are not necessary if provided the training model (file: model_pythonx.pkl)')
        else:
            tweets = self.read_file(self.dev_path)
       
        self.devset = tweets

        # Read the testset
        tweets = []
        if not os.path.exists(self.test_path):
            print('Testset file not found. You should provide this file if you want to replicate Semeval results.')
        else:
           tweets = self.read_file(self.test_path)

        self.testset = tweets
