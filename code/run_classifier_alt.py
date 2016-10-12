#########################################################################
############## Semeval - Sentiment Analysis in Twitter  #################
#########################################################################
####
####
#### Authors: Stephanie Durand
####
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
from RulesClassifier import RulesClassifier
from LexiconClassifier import LexiconClassifier
from twitter_data_reader import read_file
from PreProcess import pre_process

"""
The main mechanism for running our combined classifier.

This code is based off of the files detect_senitment.py from the aueb project folder
and the TwitterHybridClassifier.py from the hybrid_classifier project folder. Both of the 
files have been re-factored and the inputs and outputs have been changed to make 
combining the systems together. 

inputs:
tweets - a list of dictionaries that must at least contain the entries:
         'ID': <id>,
         'MESSAGE': <tweet message>

return:
a list of dictionaries of the form
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
"""
def run_classifier(tweets,d=100):
    #messages for weighted SVM classifier
    messages_test = [d['MESSAGE'] for d in tweets]
    
    #messages for rules and lexicon classifiers
    tweet_texts = [d['MESSAGE'].strip() for d in tweets]
    
    curDir = os.getcwd()
    os.chdir(join(fileDir, "hybrid_classifier"))
    
    #build the rules and lexicon classifiers
    print "Building rules and lexicon classifiers"
    rules_classifier = RulesClassifier()
    lexicon_classifier = LexiconClassifier()
    
    #pre-process tweets for rules and lexicon classifier
    tweet_tokens_list = pre_process(tweet_texts)
    
    #loading resources for building manual features vector
    os.chdir(curDir)
    os.chdir(join(fileDir, "aueb"))
    
    #tokenize all messages for weighted SVM classifier
    tokens_test = tokenize(messages_test)
    
    print "Loading resources"
    #load lexicons
    negationList, slangDictionary, lexicons, mpqa_lexicons = loadLexiconsFromFile()
	
    #load clusters
    clusters = loadClustersFromFile()
    
    #load scores
    pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores = loadScores()
    
    print "Loading glove file"
    #load Glove embeddings
    glove = loadGlove(d)
    
    print "Resources loaded"

    
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

    confidence = []
    print "Testing tweets"
    # iterate over the tweet_tokens
    for x in range(0, len(tweet_texts)):
       tweet_tokens = tweet_tokens_list[x]
       # 1. Rule-based classifier. Look for emoticons basically
       positive_score,negative_score = rules_classifier.classify(tweet_tokens)

       # 1.If any found, classify the tweet here. If none found, continue for the lexicon classifier.
       if positive_score >= 1 and negative_score == 0:
           confidence.append(np.array([0,0,1]))
           continue
       elif positive_score == 0 and negative_score <= -1:
           confidence.append(np.array([1,0,0]))
           continue

       # 2. Lexicon-based classifier
       positive_score, negative_score = lexicon_classifier.classify(tweet_tokens)

       # If in the threshold classify the tweet here. If not, continue for the ML classifier
       if positive_score >= 1 and negative_score == 0:
           confidence.append(np.array([0,0,1]))
           continue
       elif negative_score <= -2:
           confidence.append(np.array([1,0,0]))
           continue

       # 3. Machine learning based classifier - switched to use weighted SVM classifier
       result = classify(messages_test[x],
                        tokens_test[x], glove, negationList, slangDictionary, lexicons, mpqa_lexicons,
                        clusters, pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, 
                        pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, 
                        pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores, 
                        sd1, sd2, sp1, sp2)
       positive_conf = softmax(result[2])
       negative_conf = softmax(result[0])
       neutral_conf = softmax(result[1])
       confidence.append(np.array([negative_conf, neutral_conf, positive_conf]))
    
    
    os.chdir(curDir)

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
    return results, confidence

"""
Builds the feature vectors for the manually created features. 

inputs:
messages_test - list of messages/tweets
tokens_test - list of messages where each message is broken into tokens

return:
list of feature vectors for all of the messages
"""
def build_feature_vector(message, tokens, negationList, slangDictionary, lexicons, mpqa_lexicons,
        clusters, pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, 
        pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, 
        pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores):
            
    messages_test = [message]
    tokens_test = [tokens]
    if(os.path.basename(os.getcwd()) != 'aueb'):
        os.chdir(join(fileDir, "aueb"))
    #compute pos tags for all messages
    pos_tags_test = arktagger.pos_tag_list(messages_test)
    #compute pos tag bigrams
    pos_bigrams_test = getBigrams(pos_tags_test)
    #compute pos tag trigrams
    pos_trigrams_test = getTrigrams(pos_tags_test)
    
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
def build_embeddings_vector(tokens, glove):
    tokens_test = [tokens]
    embeddings_test = []
    for i in range(0, len(tokens_test)):
        embeddings_test.append(glove.findCentroid(tokens_test[i]))

    embeddings_test = np.array(embeddings_test)
    embeddings_test = regularization.regularizeHorizontally(embeddings_test)
    return embeddings_test

"""
Calls the new ML classifier 

inputs:
feature_vector - the feature vector for all of the tweets to classify
det_confidence - the confidence scores from the associated subjectivity detection classifier
classifier - the polarity classifier

return:
a 2-D array of confidence scores from the input classifier.
Each row in the array contains 3 scores [negative, neutral, positive], 
and there is a row for each message in the input list
"""
def classify(message, tokens, glove, negationList, slangDictionary, lexicons, mpqa_lexicons,
        clusters, pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, 
        pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, 
        pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores, 
        sd1, sd2, sp1, sp2):
    #build sd1 feature vector
    print "Building feature vectors"
    features_test = build_feature_vector(message, tokens, negationList, slangDictionary, lexicons, mpqa_lexicons,
        clusters, pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, 
        pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, 
        pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores)
    #build sd2 embeddings feature vector
    print "Building word embeddings feature vectors"    
    embeddings_test = build_embeddings_vector(tokens, glove)
		
    #detect sentiment
    print "Running subjectivity detection classifiers"
    sd1_confidence = sd1.decision_function(features_test)
    sd2_confidence = sd2.decision_function(embeddings_test)
    
    #classify polarity
    print "Running polarity detection classifiers"
    confidence_sp1 = classify_polarity(features_test, sd1_confidence, sp1)
    confidence_sp2 = classify_polarity(embeddings_test, sd2_confidence, sp2)
    
    #ensemble confidence scores with weight W
    W1 = 0.30
    W2 = 0.70
    
    confidence = confidence_sp1*W1 + confidence_sp2*W2
    return confidence[0]
    

def classify_polarity(feature_vector, det_confidence, classifier):

    #normalize confidence scores
    det_confidence = [softmax(conf) for conf in det_confidence]
	
    test_confidence= np.array(det_confidence)
	
    #SP features
    polarity_features = np.hstack((feature_vector, test_confidence.reshape(test_confidence.shape[0], 1)))

    #get confidence scores of sp system
    confidence = classifier.decision_function(polarity_features)
    return confidence


def print_confidence_scores(f, conf_scores):
    with open(f, 'w') as score_file:
        for conf_array in conf_scores:
            score_file.write(str(conf_array[0]) + "\t" + str(conf_array[1]) + "\t" + str(conf_array[2]) + "\n")
    


"""
Parses the input file names (as arguments), reads all of them, and detects the sentiment
of all of the combined tweets.
"""
if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print "Usage : python run_classifier.py output_file test_data_file..."
        sys.exit(0)  
    else:
        tweets = []
        output_file = sys.argv[1]
        for x in range(2, len(sys.argv)):
            #check if dataset exists
            if os.path.exists(sys.argv[x]):
                tweets += read_file(sys.argv[x])
            else:
                print sys.argv[x] + " could not be found!"
                sys.exit(0)
        print "Test files loaded"
        results = run_classifier(tweets)
        print "Writing output file"
        with open(output_file, 'w') as result_file:
            for x in range(0, len(results)):
                result_file.write(results[x][0] + "\t" + results[x][1] + "\n")
        
        print "Classifier execution complete."