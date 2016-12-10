"""
Script to perform x-fold cross-validation. This will take any number of data files and a number x as input and 
split the data such that 100/x * (x-1)%  of the data will be used for training 
and 100/x% of the input data will be used for testing.

author:Stephanie Durand
"""
import sys
import os
from os.path import join
from twitter_data_reader import read_file
from train import train
from train import train_classifier3
from train_aueb import train as aueb_train
from aueb.detect_sentiment import main as detect_sentiment
import argparse
from run_classifier_rules_pipe import run_classifier
fileDir = os.path.dirname(__file__)
"""
Creates the data folds to be used for cross-validation

input:
data_list - list containing all of the data to be used for training and testing
numFolds - the number of folds to create out of the data

output - list of data lists. There should be numFolds equally-sized lists in the returned list 
"""
def fold_data(data_list, numFolds):
        # split the data into x "folds"
        foldSize = int(round(len(data_list)/float(numFolds)))
        folds = []
        index = 0
        while index < len(data_list):
            endIndex = index + foldSize
            if(index + foldSize < len(data_list)):
                fold = data_list[index:endIndex]
                folds.append(fold)
            else:
                if len(folds) == numFolds:
                    folds[-1].append(data_list[index:])
                else:
                    fold = data_list[index:]
                    folds.append(fold)
            index = endIndex
        return folds
 
"""
Trains and tests using the new combined classifier

inputs:
training - a list of dictionaries for each message in the training set. 
    The dictionary should be in the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
     }
test - a list of dictionaries for each message in the test set.
The dictionary will likely be of the same format as the training set, but can be missing the sentiment entry.

d - the number of dimensions to use for the Glove data

outputs:
a list of dictionaries for each message in the test set of the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>
    }
"""
def run_combined_classifier(training, test, d):
        #train the classifier
        train(training, d)
        print "Training complete."
        #run the classifier
        all_results = run_classifier(test, d)
        print "Classification complete"
        return all_results[0]
  
    
"""
Trains and tests using the original weightedSVM classifier

inputs:
training - a list of dictionaries for each message in the training set. 
    The dictionary should be in the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
     }
test - a list of dictionaries for each message in the test set.
The dictionary will likely be of the same format as the training set, but can be missing the sentiment entry.

d - the number of dimensions to use for the Glove data

outputs:
a list of dictionaries for each message in the test set of the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>
    }
"""
def run_weightedSVM_classifier(training, test, d):
    curDir = os.getcwd()
    aueb_train(training, d)
    print "Training complete"
    
    os.chdir(fileDir +"aueb")
    
    #create test data
    message_tests = {}
    for test_tweet in test:
        message_tests[test_tweet['MESSAGE']] = test_tweet['ID']

    #run classifier
    results = detect_sentiment(message_tests)
    os.chdir(curDir)
    return_results = []
    for tweet_id, polarity in results:
      return_results.append({'ID': tweet_id, 'SENTIMENT': polarity})
    return return_results
    
"""
Trains and tests using the original pipeline classifier

inputs:
training - a list of dictionaries for each message in the training set. 
    The dictionary should be in the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
     }
test - a list of dictionaries for each message in the test set.
The dictionary will likely be of the same format as the training set, but can be missing the sentiment entry.

outputs:
a list of dictionaries for each message in the test set of the following format:
    {
         'ID': <id>,
         'SENTIMENT': <sentiment>
    }
"""
def run_pipeline_classifier(training, test):

    #train
    #switch directory because all of the file paths are hardcoded in the hybrid classifer project :(
    curDir = os.getcwd()
    os.chdir(join(fileDir, "hybrid_classifier"))
    
    #train the new ML classifier for the pipeline
    classifier = train_classifier3(training)
    print "Trained pipeline classifier"
    
    #test
    #run hybrid pipeline classifier
    print "Running hybrid classifier"
    tweet_texts = [tweet['MESSAGE'].strip() for tweet in test]
    
    predictions = classifier.classify_batch(tweet_texts)
    
    #switch back to current directory
    os.chdir(curDir)
    results = []
    for i in range(0,len(predictions)):
      results.append({'ID': test[i]['ID'], 'SENTIMENT': predictions[i][0]})
    return results
    
    
"""    
Parses the input file names (as arguments), reads all of them, splits the data into the correct ratio of 
training to test data, trains using the training data and classifies the test data using the specfied classifier.
The classifier can either be the original weightedSVM classifier, the original pipeline classifier, or
our new classifier that combines the two.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, choices=['VCU', 'weightedSVM', 'pipeline'],
                    help='specifies that the classifier to use for cross-fold validation', default='VCU')
    parser.add_argument('--x', type=int, default=10, help="the number of folds to use in cross-fold validation")
    parser.add_argument('--d', type=int, choices=[25, 50, 100, 200], default=100, 
        help="the number of dimensions to use for the glove word vector representations.")
    parser.add_argument('output_file', type=str, nargs=1, help="the name of the output file")
    parser.add_argument('files', type=str, nargs='+', help="the data files to use for cross-validation")
    args = parser.parse_args()

    tweets = []
    for filename in args.files:
        #check if dataset exists
        if os.path.exists(filename):
            #read all of the tweets from the file and add them to the list 
            #of dictionaries representing the tweets
            tweets += read_file(filename)
        else:
            #notify of an error finding the file
            print filename + " could not be found!"
            sys.exit(0)
    print "Data files loaded"

    folds = fold_data(tweets, args.x)
    results = []
    for foldNum in range(0,len(folds)):
        training = []
        test = []
        for index in range(0, len(folds)):
            if index == foldNum:
                test = folds[index]
            else:
                training.extend(folds[index])
                
        if(args.classifier=='VCU'):
            results.extend(run_combined_classifier(training, test, args.d))
        elif(args.classifier=='weightedSVM'):
            results.extend(run_weightedSVM_classifier(training, test, args.d))
        else:
            results.extend(run_pipeline_classifier(training, test))
            
    with open(args.output_file[0], "w") as outfile:    
        for entry in results:
            outfile.write('{}\t{}\n'.format(entry['ID'], entry['SENTIMENT']))
            
