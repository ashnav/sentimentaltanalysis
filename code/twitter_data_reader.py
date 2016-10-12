#########################################################################
############## Semeval - Sentiment Analysis in Twitter  #################
#########################################################################
####
#### Authors: Stephanie Durand
#### 

"""
Reads a file in the format of the training and test data from the SemEval 2016 data set.
Each line should be formatted as: <tweet_id>\t<sentiment>\t<tweet>.
"""  
def read_file(f):
   tweets = []
   with open(f, 'r') as input_file:
      for line in input_file:
        tweet_line = line.split('\t')
        #check for invalid input
        if len(tweet_line) != 3:
            print('Error reading file . Each line must contain 3 arguments, <tweet_id> <sentiment> <tweet>. Line: ', line)
        tweet = {}
        #parse the first item as the tweet id
        tweet['ID'] = tweet_line[0]
        #parse the second item as the sentiment
        sentiment = tweet_line[1]
        tweet['SENTIMENT'] = sentiment
        #parse the third item as the tweet message
        tweet['MESSAGE'] = tweet_line[2]
   return tweets 


