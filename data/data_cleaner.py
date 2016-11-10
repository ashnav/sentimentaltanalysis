import sys
import os
"""
Reads a file in the format of the training and test data from the SemEval 2013 or 2016 data set.
Each line should be formatted as: <tweet_id>\t<sentiment>\t<tweet> for 2016 data or 
<tweet_id>\t<other_id>\t<sentiment>\t<tweet> for 2013 data.
inputs:
f - input file in the correct format

returns:
tweets - a list of dictionaries containing tweet information for only the available tweets.
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
        line_len = len(tweet_line)
        #check for invalid input
        if line_len != 3 and line_len !=4:
            print 'Error reading file . Each line must contain 3 or 4 arguments separated by tabs, <tweet_id> <sentiment> <tweet>. Line: ' + line 
        tweet = {}
        sentiment_index = 1;
        if(line_len == 4):
            sentiment_index = 2
        message_index = 2
        if( line_len == 4):
            message_index = 3
        #parse the first item as the tweet id
        tweet['ID'] = tweet_line[0]
        #parse the sentiment
        sentiment = tweet_line[sentiment_index]
        tweet['SENTIMENT'] = "neutral" if (sentiment == "objective" or sentiment == "objective-OR-neutral") else sentiment
        #parse the tweet message
        message = tweet_line[message_index]
        tweet['MESSAGE'] = message
        #add the dictionary to the list
        if(message.strip() != "Not Available"):
            tweets.append(tweet)
        else:
            print "Not Available"
    return tweets

"""
Writes a file in the format of the training  data from the SemEval 2016 data set.
Each line will be formatted as: <tweet_id>\t<sentiment>\t<tweet> 
inputs:
o - desired output file location
tweets - a list of dictionaries containing tweet information to be written to the file.
      Each dictionary will be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
"""
def write_file(o,tweets):
    with open(o, "w") as out:
        for tweet in tweets:
            out.write(tweet['ID'] + "\t" + tweet['SENTIMENT'] + "\t" + tweet['MESSAGE'])

"""
Parses the input file names (as arguments), reads all of them, and 
writes the tweets in the 2016 training data format. This will remove any of the 
"Not Available" tweets and set all "objective" and "objective-OR-neutral" sentiments to "neutral".
This will also change 2013 formatted data to the 2016 format.

"""
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print "Usage : python data_cleaner.py input_file..."
        sys.exit(0)  
    else:

        #read all of the arguments as input files
        for x in range(1, len(sys.argv)):
            tweets = []
            input_file = sys.argv[x]
            #set the output file to an altered version of the input file name
            output_file = input_file.replace("downloaded", "clean")
            #check if dataset exists
            if os.path.exists(input_file):
                #read all of the tweets from the file 
                print "Reading input file"
                tweets = read_file(input_file)
                #write all of the available tweets to the output file
                print "Writing output file"
                write_file(output_file, tweets)
            else:
                #notify of error finding an input file
                print sys.argv[x] + " could not be found!"