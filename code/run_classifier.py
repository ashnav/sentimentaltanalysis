# 
# As our project was based mostly on combining techniques with existing code, there was not a lot of code, so 
# Stephanie Durand wrote or adapted all of the code for the project phase I.
#
# This file is the main mechanism for running our combined classifier.
#
# This code is based off of the files detect_senitment.py from the aueb project folder
# and the TwitterHybridClassifier.py from the hybrid_classifier project folder. Both of the 
# files have been re-factored and the inputs and outputs have been changed to be able to
# combine the systems together. 
#
# Adapted from code written by:
# Stavros Giorgis, Apostolos Rousas, John Pavlopoulos, 
# Prodromos Malakasiotis and Ion Androutsopoulos
# and 
# Pedro Paulo Balage Filho and Lucas Avanco
#
# Adapted by: Stephanie Durand
#
import sys
import os
from os.path import join
fileDir = os.path.dirname(__file__)
sys.path.append(join(fileDir, "aueb"))
sys.path.append(join(fileDir, "hybrid_classifier"))
from train import softmax
from postaggers import arktagger
from utilities import *
from features import features
import numpy as np
import regularization
from TwitterHybridClassifier import TwitterHybridClassifier
from twitter_data_reader import read_file
from PreProcess import pre_process

"""
This function will run all of the classifiers and combine their confidence scores
to predict the sentiment of the provided tweets.  

inputs:
tweets - a list of dictionaries that must at least contain the entries:
        {
         'ID': <id>,
         'MESSAGE': <tweet message>
        }
        (The dictionaries can have other entries that will be ignored.)

return:
a tuple containing the following data at the noted index
 0 - a list of dictionaries of the form
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
     with the sentiments predicted from the ensemble of classifiers,
 1 - the confidence scores from SP1, 
 2 - the confidence scores from SP2, 
 3 - the confidence scores from the hybrid classifier pipeline

"""
def run_classifier(tweets, d):
    messages_test = [tweet['MESSAGE'] for tweet in tweets]
    curDir = os.getcwd()
    
    #switch directory because all of the file paths are hardcoded in the hybrid_classifier project :(
    os.chdir(join(fileDir, "hybrid_classifier"))
    #run hybrid pipeline classifier
    print "Running hybrid classifier"
    tweet_texts = [tweet['MESSAGE'].strip() for tweet in tweets]
    confidence_pipe = run_hybrid_classifier(tweet_texts)

    #tokenize all messages
    tokens_test = tokenize(messages_test)
    
    #switch directory back to folder containing this file
    os.chdir(curDir)
    #load SD classifiers
    print "Loading subjectivity detection classifiers"
    with open(join(join(fileDir, 'aueb'),'resources/sd_models.pkl'), 'rb') as input:
        sd1 = pickle.load(input)
        sd2 = pickle.load(input)
        
    #load SP classifiers
    print "Loading polarity detection classifiers"
    with open(join(join(fileDir, 'aueb'),'resources/sp_models.pkl'), 'rb') as input:
        sp1 = pickle.load(input)
        sp2 = pickle.load(input)
        
    #switch directory because all of the file paths are hardcoded in the aueb project :(
    os.chdir(join(fileDir, "aueb"))
    
    #build sd1 feature vector
    print "Building feature vectors"
    features_test = build_feature_vector(messages_test, tokens_test)
    
    #build sd2 embeddings feature vector
    print "Building word embeddings feature vectors"
    embeddings_test = build_embeddings_vector(tokens_test)
		
    #detect sentiment
    print "Running subjectivity detection classifiers"
    sd1_confidence = sd1.decision_function(features_test)
    sd2_confidence = sd2.decision_function(embeddings_test)
    
    #classify polarity
    print "Running polarity detection classifiers"
    confidence_sp1 = classify_polarity(features_test, sd1_confidence, sp1)
    confidence_sp2 = classify_polarity(embeddings_test, sd2_confidence, sp2)
    
    #switch directory back to folder containing this file
    os.chdir(curDir)
    
    #combine confidence scores with weights W1, W2, W3
    W1 = 0.34
    W2 = 0.58
    W3 = 0.08
    
    confidence = confidence_sp1*W1 + confidence_sp2*W2 + confidence_pipe*W3

    #get final prediction
    prediction = [np.argmax(x)-1 for x in confidence]
    prediction = np.array(prediction)

    results = []
    for i in range(0, prediction.shape[0]):
        if prediction[i] == -1:
            pol = "negative"
        elif prediction[i] == 0:
            pol = "neutral"
        else:
            pol = "positive"
        results.append((tweets[i]['ID'], pol))
    print "Classification complete"
    return (results, confidence_sp1, confidence_sp2, confidence_pipe)

"""
Builds the feature vectors for the manually created features. 

inputs:
messages_test - list of messages/tweets
tokens_test - list of messages where each message is broken into tokens

return:
list of feature vectors for all of the messages
"""
def build_feature_vector(messages_test, tokens_test):
    #compute pos tags for all messages
    pos_tags_test = arktagger.pos_tag_list(messages_test)
    #compute pos tag bigrams
    pos_bigrams_test = getBigrams(pos_tags_test)
    #compute pos tag trigrams
    pos_trigrams_test = getTrigrams(pos_tags_test)

    #load scores
    pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores = loadScores()
	
    #load lexicons
    negationList, slangDictionary, lexicons, mpqa_lexicons = loadLexiconsFromFile()
	
    #load clusters
    clusters = loadClustersFromFile()
		
    print "Resources loaded"
    
    features_test = features.getFeatures(
                                messages_test, 
                                tokens_test, 
                                pos_tags_test, 
                                slangDictionary, 
                                lexicons, 
                                mpqa_lexicons, 
                                pos_bigrams_test, 
                                pos_trigrams_test, 
                                pos_bigrams_scores_negative, 
                                pos_bigrams_scores_positive, 
                                pos_trigrams_scores_negative, 
                                pos_trigrams_scores_positive, 
                                pos_tags_scores_negative, 
                                pos_tags_scores_positive, 
                                mpqaScores, 
                                negationList, 
                                clusters, 
                                pos_bigrams_scores_neutral, 
                                pos_trigrams_scores_neutral, 
                                pos_tags_scores_neutral)
    
    features_test = regularization.regularize(features_test)
    return features_test


"""
Builds the feature vectors for the word embeddings. 

inputs:
tokens_test - list of messages where each message is broken into tokens
d - the size of the word embeddings file (25, 50, 100, 200) 
    warning: a d value of 200 caused my machine to run out of memory.

return:
list of feature vectors for all of the messages built from the word embeddings
"""
def build_embeddings_vector(tokens_test, d=100):
    
    #load Glove embeddings
    glove = loadGlove(d)
    
    embeddings_test = []
    for i in range(0, len(tokens_test)):
        embeddings_test.append(glove.findCentroid(tokens_test[i]))

    embeddings_test = np.array(embeddings_test)
    embeddings_test = regularization.regularizeHorizontally(embeddings_test)
    return embeddings_test

"""
Calls the provided polarity classifier. 

inputs:
feature_vector - the feature vector for all of the tweets to classify
det_confidence - the confidence scores from the associated subjectivity detection classifier
classifier - the polarity classifier

return:
a 2-D array of confidence scores from the input classifier.
Each row in the array contains 3 scores [negative, neutral, positive], 
and there is a row for each message in the input list
"""
def classify_polarity(feature_vector, det_confidence, classifier):

    #normalize confidence scores
    det_confidence = [softmax(conf) for conf in det_confidence]
	
    test_confidence= np.array(det_confidence)
	
    #SP features
    polarity_features = np.hstack((feature_vector, test_confidence.reshape(test_confidence.shape[0], 1)))

    #get confidence scores of sp system
    confidence = classifier.decision_function(polarity_features)

    #normalize new confidence scores
    for i in range(0, confidence.shape[0]):
        for j in range(0, confidence.shape[1]):
            confidence[i][j] = softmax(confidence[i][j])
    return confidence

"""
Calls the hybrid classifier.

inputs:
tweet_texts - a list of tweets/messages

return:
a 2-D array of confidence scores from the hybrid classifier.
Each row in the array contains 3 scores [negative, neutral, positive], 
and there is a row for each message in the input list
"""
def run_hybrid_classifier(tweet_texts):
    hybrid_classifier = TwitterHybridClassifier()
    confidence = get_hybrid_confidence(tweet_texts, hybrid_classifier)
    return confidence
    
"""
Calculates the confidence scores for the hybrid classifier

inputs:
tweet_texts - a list of tweets (messages)
hybrid_classifier - a pre-trained TwitterHybridClassifier

return:
a 2-D array of confidence scores from the hybrid classifier.
Each row in the array contains 3 scores [negative, neutral, positive], 
and there is a row for each message in the input list
"""
def get_hybrid_confidence(tweet_texts, hybrid_classifier):
    # 0. Pre-process the tweets (tokenization, tagger, normalizations)
    tweet_tokens_list = []

    if len(tweet_texts) == 0:
       return []

    print 'Preprocessing the test data'
    # pre-process the tweets
    tweet_tokens_list = pre_process(tweet_texts)


    confidence = []
    print "Testing tweets with hybrid classifier"
    # iterate over the tweet_tokens
    for index, tweet_tokens in enumerate(tweet_tokens_list):
       # 1. Rule-based classifier.
       positive_score,negative_score = hybrid_classifier.rules_classifier.classify(tweet_tokens)

       # 1.If any found, classify the tweet here. If none found, continue for the lexicon classifier.
       if positive_score >= 1 and negative_score == 0:
           confidence.append(np.array([0,0,1]))
           continue
       elif positive_score == 0 and negative_score <= -1:
           confidence.append(np.array([1,0,0]))
           continue

       # 2. Lexicon-based classifier
       positive_score, negative_score = hybrid_classifier.lexicon_classifier.classify(tweet_tokens)

       # If in the threshold classify the tweet here. If not, continue for the ML classifier
       if positive_score >= 1 and negative_score == 0:
           confidence.append(np.array([0,0,1]))
           continue
       elif negative_score <= -2:
           confidence.append(np.array([1,0,0]))
           continue

       # 3. Machine learning based classifier
       result = hybrid_classifier.ml_classifier.classify(tweet_tokens)
       positive_conf = softmax(result['positive'])
       negative_conf = softmax(result['negative'])
       neutral_conf = softmax(result['neutral'])
       confidence.append(np.array([negative_conf, neutral_conf, positive_conf]))
    confidence = np.array(confidence)
    return confidence

"""
Helper function for printing out the confidence scores for 
use in the tuning script.

inputs:
f - the file path for printing the confidence scores
conf_scores - the confidence scores for a single classifier (SP1, SP2, or the pipeline)
"""
def print_confidence_scores(f, conf_scores):
    with open(f, 'w') as score_file:
        for conf_array in conf_scores:
            score_file.write(str(conf_array[0]) + "\t" + str(conf_array[1]) + "\t" + str(conf_array[2]) + "\n")
    

"""
Parses the input file names (as arguments), reads all of them, and detects the sentiment
of all of the combined tweets.
"""
if __name__ == "__main__":
    #check for incorrect number of arguments
    if(len(sys.argv) < 4):
        print "Usage : python run_classifier.py <# of dimensions for word embeddings> <output_file> <test_data_file>..."
        sys.exit(0)  
    else:
        tweets = []
        #set the number of dimensions for the word embeddings vectors
        d = int(sys.argv[1])
        #set the output file to the first argument
        output_file = sys.argv[2]
        #read all of the remaining arguments as input testing files
        for x in range(3, len(sys.argv)):
            #check if dataset exists
            if os.path.exists(sys.argv[x]):
                #read all of the tweets from the file and add them to the list 
                #of dictionaries representing the tweets
                tweets += read_file(sys.argv[x])
            else:
                #notify of error finding an input file
                print sys.argv[x] + " could not be found!"
                sys.exit(0)
        print "Test files loaded"
        print "Beginning classification"
        all_results = run_classifier(tweets,d)
        results = all_results[0]
        print "Writing confidence score files"
        conf_sp1 = "code/tuning/conf_sp1_new"
        print_confidence_scores(conf_sp1, all_results[1])
        conf_sp2 = "code/tuning/conf_sp2_new"
        print_confidence_scores(conf_sp2, all_results[2])
        conf_pipe = "code/tuning/conf_pipe_new"
        print_confidence_scores(conf_pipe, all_results[3])
        
        print "Writing output file"
        with open(output_file, 'w') as result_file:
            for x in range(0, len(results)):
                result_file.write(results[x][0] + "\t" + results[x][1] + "\n")
        
        print "Classifier execution complete."
