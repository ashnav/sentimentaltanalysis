"""
Script to perform x-fold cross-validation. This will take any number of data files and a number x as input and 
split the data such that 100/x * (x-1)%  of the data will be used for training 
and 100/x% of the input data will be used for testing.

author:Stephanie Durand
"""
import sys
import os
from twitter_data_reader import read_file
from train import train
from run_classifier import run_classifier

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
                if len(folds) == x:
                    folds[-1].append(data_list[index:])
                else:
                    fold = data_list[index:]
                    folds.append(fold)
            index = endIndex
        return folds
    
"""
Parses the input file names (as arguments), reads all of them, splits the data into the correct ratio of 
training to test data, trains using the training data and classifies the test data.
"""

if __name__ == "__main__":
    #checks for invalid input arguments
    if(len(sys.argv) < 2):
        print "Usage : python cross_validate.py x d train_dataset_path..."
        sys.exit(0)  
    else:
        tweets = []
        x = int(sys.argv[1])
        d= int(sys.argv[2])
        for arg in range(3, len(sys.argv)):
            #check if dataset exists
            if os.path.exists(sys.argv[arg]):
                #read all of the tweets from the file and add them to the list 
                #of dictionaries representing the tweets
                tweets += read_file(sys.argv[arg])
            else:
                #notify of an error finding the file
                print sys.argv[arg] + " could not be found!"
                sys.exit(0)
        print "Data files loaded"
        
        folds = fold_data(tweets, x)
        
        for foldNum in range(0,len(folds)):
            training = []
            test = []
            for index in range(0, len(folds)):
                if index == foldNum:
                    test = folds[index]
                else:
                    training.extend(folds[index])
        
            train(training, d)
            print "Training complete."
            all_results = run_classifier(test, d)
            print "Classification complete"
            results = all_results[0]
            filename = "cross_validate" + str(foldNum) + ".out"
            with open(filename, "w") as outfile:
                for result in results:
                    outfile.write(result[0] + '\t' + result[1] + '\n')
            print "Execution complete. Output file written as " + filename
