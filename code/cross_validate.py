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
from run_classifier import run_classifier
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
Trains and tests using the combined classifier
"""
def run_combined_classifier(training, test, foldNum, d):
        #train the classifier
        train(training, d)
        print "Training complete."
        #run the classifier
        all_results = run_classifier(test, d)
        print "Classification complete"
        results = all_results[0]
        filename = "combined_results_fold" + str(foldNum) + ".out"
        with open(filename, "w") as outfile:
            for result in results:
                outfile.write(result[0] + '\t' + result[1] + '\n')
        print "Execution complete. Output file written as " + filename
    
"""
Trains and tests using the original weightedSVM classifier
"""
def run_weightedSVM_classifier(training, test, foldNum, d):
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
    #write results
    with open(join(fileDir,"../data/aueb_results_fold" + str(foldNum)), "w") as results_out:
       for tweet_id, polarity in results:
          results_out.write(tweet_id +"\t" + polarity + "\n")
    
"""
Trains and tests using the original pipeline classifier
"""
def run_pipeline_classifier(training, test, foldNum):
    print "in run pipeline"
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
    #write results
    with open(join(fileDir,"../data/pipeline_results_fold" + str(foldNum)), "w") as results_out:
       for i in range(0,len(predictions)):
          results_out.write(test[i]['ID'] +"\t" + predictions[i][0] + "\n")
    
    
"""    
Parses the input file names (as arguments), reads all of them, splits the data into the correct ratio of 
training to test data, trains using the training data and classifies the test data.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, choices=['VCU', 'weightedSVM', 'pipeline'],
                    help='specifies that the classifier to use for cross-fold validation', default='VCU')
    parser.add_argument('--x', type=int, default=10, help="the number of folds to use in cross-fold validation")
    parser.add_argument('--d', type=int, choices=[25, 50, 100, 200], default=100, 
        help="the number of dimensions to use for the glove word vector representations.")
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

    for foldNum in range(0,len(folds)):
        training = []
        test = []
        for index in range(0, len(folds)):
            if index == foldNum:
                test = folds[index]
            else:
                training.extend(folds[index])
        
        if(args.classifier=='VCU'):
            run_combined_classifier(training, test, foldNum ,args.d)
        elif(args.classifier=='weightedSVM'):
            run_weightedSVM_classifier(training, test, foldNum, args.d)
        else:
            run_pipeline_classifier(training, test, foldNum)
            