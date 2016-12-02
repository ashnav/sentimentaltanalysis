"""
Script to create feature vectors for the specified data and create files in the Attribute-Relation File Format (ARFF).
The file(s) will be used in weka for calculating the information gain of each of the features to determine which ones 
we should keep.

author:Stephanie Durand
"""
import os
import sys
from os.path import join
fileDir = os.path.dirname(__file__)
sys.path.append(join(fileDir, "../"))
sys.path.append(join(fileDir, "../aueb"))
sys.path.append(join(fileDir, "../hybrid_classifier"))
from twitter_data_reader import read_file
from PreProcess import pre_process
from utilities import *
from train import build_feature_vector
import MachineLearningClassifier 


def printWeightedSVMfeatures(tweets):
        
        #make lists from the dictionary values for use in the Weighted SVM code
        raw_labels_train = [tweet['SENTIMENT'] for tweet in tweets]
        messages_train = [tweet['MESSAGE'] for tweet in tweets]

        #labels for polarity detection (3 categories)
        labels_train = [0 if x == "neutral" else -1 if x == "negative" else 1 for x in raw_labels_train]
        labels_train = np.array(labels_train)
        
        #tokenize all messages for weighted SVMs
        tokens_train = tokenize(messages_train)
        #switch directory because all of the file paths are hardcoded in the aueb project as well :(
        curDir = os.getcwd()
        os.chdir(join(fileDir, "../aueb"))
        print "Building feature vectors"
        features = build_feature_vector(messages_train, tokens_train, labels_train)
        os.chdir(curDir)
        print "Printing feature vectors"
        with open("data/weightedSVM_features.arff", "w") as outfile:
            message_num = 0
            for message_features in features:
                features_str = ""
                for feature in message_features:
                    features_str += str(feature) + ","
                features_str += tweets[message_num]['SENTIMENT']
                outfile.write(features_str + "\n")
                message_num+=1


def printPipelineFeatures(tweets):
    messages = [(tweet['MESSAGE'], tweet['SENTIMENT']) for tweet in tweets]
        
    print ('Preprocessing the training data')
    tweet_messages = [tweet_message for tweet_message,label in messages]
    tweet_labels = [label for tweet_message,label in messages]

    curDir = os.getcwd()
    os.chdir(join(fileDir, "../hybrid_classifier"))
    # preproces all the tweet_messages (Tokenization, POS and normalization)
    tweet_tokens = pre_process(tweet_messages)

    # compile a trainset with tweek_tokens and labels (positive,
    # negative or neutral)

    trainset = [(tweet_tokens[i],tweet_labels[i]) for i in range(len(messages))]

    #initialize the classifier
    mlClassifier = MachineLearningClassifier.MachineLearningClassifier(trainset)
    featureNames = ['pos_lexicon','neg_lexicon','has_char_ngrams','num_all_caps',
    'has_punct_ngrams','num_T','num_,','num_!','num_#','num_$','num_&','num_L',
    'num_M','num_N','num_O','num_@','num_A','num_D','num_E','num_G','num_X',
    'num_Y','num_Z','num_^','num_P','num_R','num_S','num_U','num_V','num_~']

    with open("../../data/pipeline_features.arff", "w") as outfile:
        for feature in featureNames:
            outfile.write(feature.encode('utf8') + "\n")
        #create the feature vector for each message
        for tweet_token, label in trainset:
            features = mlClassifier.extract_features(tweet_token)
            instance_features = ""
            for feature in featureNames:
                instance_features += str(features[feature]) +","
            instance_features += label
            outfile.write(instance_features + "\n")       
    os.chdir(curDir)
    
    

if __name__ == "__main__":
    #checks for invalid input arguments
    if(len(sys.argv) < 2):
        print "Usage : python weka_data_builder.py dataset_path..."
        sys.exit(0)  
    else:
        tweets = []
        for arg in range(1, len(sys.argv)):
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
        
        printWeightedSVMfeatures(tweets)    