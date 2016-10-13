#Authors: Stephanie Durand
#
# This file contains utility methods for reading files in the format provided 
# by the SemEval 2017 Task 4 organizers.
#

"""
Reads a file in the format of the training and test data from the SemEval 2016 data set.
Each line should be formatted as: <tweet_id>\t<sentiment>\t<tweet>.
inputs:
f - input file in the correct format

returns:
tweets - a list of dictionaries containing tweet information.
      Each dictionary will be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
"""  
def read_file(f):
    tweets = []
    with open(f, 'r') as input_file:
      for line in input_file:
        tweet_line = line.split('\t')
        #check for invalid input
        if len(tweet_line) != 3:
            print 'Error reading file . Each line must contain 3 arguments separated by tabs, <tweet_id> <sentiment> <tweet>. Line: ' + line 
        tweet = {}
        #parse the first item as the tweet id
        tweet['ID'] = tweet_line[0]
        #parse the second item as the sentiment
        sentiment = tweet_line[1]
        tweet['SENTIMENT'] = sentiment
        #parse the third item as the tweet message
        tweet['MESSAGE'] = tweet_line[2]
        #add the dictionary to the list
        tweets.append(tweet)
    return tweets 


