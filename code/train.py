#
# This file combines the training steps from the "Weighted SVMs" and "Hybrid Classifier" approaches.
# This is based off of the train.py code from the aueb folder and the run_semeval_classifier.py code
# from the hybrid_classifier folder.
# The original code from train.py forced a single file at a time as input and would use its own reader, 
# forcing us to read the same file multiple times. Our version is an adaptation of the original 
# with extra input flexibility and major reafactorization to make the code easier to 
# understand and reduce code duplication. 
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
from twitter_data_reader import read_file
from postaggers import arktagger
from utilities import *
from features import features
from classifiers import SVM
import numpy as np
import regularization
from embeddings import GloveDictionary
from TwitterHybridClassifier import TwitterHybridClassifier
import math

"""
This function will train all of the machine learning classifiers and save them off so 
that they can be used for testing.

inputs:
tweets - a list of dictionaries containing tweet information.
      Each dictionary should be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
d - the number of dimensions for the word embeddings vectors
      
"""
def train(tweets, d):
    print "System training started"
    
#    #switch directory because all of the file paths are hardcoded in the hybrid classifer project :(
    curDir = os.getcwd()
#    os.chdir(join(fileDir, "hybrid_classifier"))
#    
#    #train the new ML classifier for the pipeline
#    train_classifier3(tweets)
#    print "Trained new hybrid classifier"
#    
#    #switch back to current directory
#    os.chdir(curDir)
    
    print "Training the SVMs"
    #make lists from the dictionary values for use in the Weighted SVM code
    raw_labels_train = [tweet['SENTIMENT'] for tweet in tweets]
    messages_train = [tweet['MESSAGE'] for tweet in tweets]
	
    #labels for subjectivity detection (2 categories)
    temp_labels_train = [0 if x == "neutral" else 1 for x in raw_labels_train]
    #labels for polarity detection (3 categories)
    labels_train = [0 if x == "neutral" else -1 if x == "negative" else 1 for x in raw_labels_train]
	
    #convert labels to numpy arrays
    temp_labels_train = np.array(temp_labels_train)
    labels_train = np.array(labels_train)
	
    #tokenize all messages for weighted SVMs
    tokens_train = tokenize(messages_train)
   
    #switch directory because all of the file paths are hardcoded in the aueb project as well :(
    os.chdir(join(fileDir, "aueb"))

    #build the feature vector for manual technique (SD1)
    features_train = build_feature_vector(messages_train, tokens_train, labels_train)
    print "built the manual feature vector"
	
    #build the feature vector for the word embeddings technique (SD2)
    embeddings_train = build_embeddings_vector(tokens_train, d)
    print "built the word embeddings feature vector"
   
	
    #Penalty parameter C of the error term of the SP systems
    C1 = 0.003410871889693192
    C2 = 7.396183688299606
   
    #Subjectivity Detection
    #train sd1
    sd1 = train_sd1(features_train, temp_labels_train, C1)
    print "trained sd1"
   
    #train sd2
    sd2 = train_sd2(embeddings_train, temp_labels_train, C2)
    print "trained sd2"
   
    #Sentiment Polarity 
    #train SP classifiers
    sp1 = train_sp1(features_train, temp_labels_train, labels_train, C1)
    print "trained sp1"
    sp2 = train_sp2(embeddings_train, temp_labels_train, labels_train, C2)
    print "trained sp1"

    #save trained models
    saveModels(sd1, sd2, sp1, sp2)
    print "saved sd1, sd2, sp1, and sp2"
    
    #switch file directory back
    os.chdir(curDir)

    print "Training complete."

"""
Build features for manually-created feature vector

inputs: 
messages_train - list of tweets to use for training (just the messages)
tokens_train - list of lists of tokens for each message in messages_train 
labels_train - list of labels for each of the messages (in the same order as the messages)

returns:
a list of the feature vectors for all of the input training tweets
"""
def build_feature_vector(messages_train, tokens_train, labels_train):

    #load word clusters
    clusters = loadClusters()
	
    #load Lexicons
    negationList, slangDictionary, lexicons, mpqa_lexicons = loadLexicons()

    #compute pos tags for all messages
    pos_tags_train = arktagger.pos_tag_list(messages_train)
	
    #compute pos tag bigrams
    pos_bigrams_train = getBigrams(pos_tags_train)

    #compute pos tag trigrams
    pos_trigrams_train = getTrigrams(pos_tags_train)

    #get the unique pos bigrams from training set
    unique_pos_tags = getPosTagsSet(pos_tags_train)
    unique_bigrams = getBigramsSet(pos_bigrams_train)
    unique_trigrams = getTrigramsSet(pos_trigrams_train)

    #compute POS tag scores
    pos_tags_scores_neutral = posTagsScore(unique_pos_tags, 0, pos_tags_train, labels_train)
    pos_tags_scores_positive = posTagsScore(unique_pos_tags, 1, pos_tags_train, labels_train)
    pos_tags_scores_negative = posTagsScore(unique_pos_tags, -1, pos_tags_train, labels_train)
	   
    pos_bigrams_scores_neutral = posBigramsScore(unique_bigrams, 0, pos_bigrams_train, labels_train)
    pos_bigrams_scores_positive = posBigramsScore(unique_bigrams, 1, pos_bigrams_train, labels_train)
    pos_bigrams_scores_negative = posBigramsScore(unique_bigrams, -1, pos_bigrams_train, labels_train)

    pos_trigrams_scores_neutral = posTrigramsScore(unique_trigrams, 0, pos_trigrams_train, labels_train)
    pos_trigrams_scores_positive = posTrigramsScore(unique_trigrams, 1, pos_trigrams_train, labels_train)
    pos_trigrams_scores_negative = posTrigramsScore(unique_trigrams, -1, pos_trigrams_train, labels_train)
	
    #compute mpqa scores
    mpqaScores = getScores(mpqa_lexicons, messages_train, labels_train, neutral=True)
	
    #save scores and other resources for future use
    savePosScores(pos_tags_scores_neutral, pos_tags_scores_positive, pos_tags_scores_negative, pos_bigrams_scores_neutral, pos_bigrams_scores_positive, pos_bigrams_scores_negative, pos_trigrams_scores_neutral, pos_trigrams_scores_positive, pos_trigrams_scores_negative, mpqaScores)
    #save lexicons
    saveLexicons(negationList, slangDictionary, lexicons, mpqa_lexicons)
    #save clusters
    saveClusters(clusters)
    
    #build the manual feature vector
    features_train = features.getFeatures(
                                          messages_train,
                                          tokens_train,
                                          pos_tags_train,
                                          slangDictionary,
                                          lexicons,
                                          mpqa_lexicons,
                                          pos_bigrams_train,
                                          pos_trigrams_train,
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
    #regularize features
    features_train = regularization.regularize(features_train)
	
    #return feature vector
    return features_train

"""
Builds the feature vector for the word embeddings approach

inputs:
tokens_train - list of lists of tokens for each of the input tweets
d - the number of dimensions for the word embeddings vectors. 
    This will determine which GloVe file will be used to create the feature vector.

returns:
a list of the feature vectors based on word embeddings for all of the input training tweets
"""
def build_embeddings_vector(tokens_train, d=100):
    #load Glove embeddings
    #the original model used a d value of 200, but my computer couldn't handle it. I kept running out of memory.
    glove = GloveDictionary.Glove(d)

    #save Glove embeddings for future use
    saveGlove(glove)
    
    #word embeddings features
    print "Building glove feature vectors"
    embeddings_train = []
    #for each message in tokens_train :
    for i in range(0, len(tokens_train)):
        embeddings_train.append(glove.findCentroid(tokens_train[i]))
    embeddings_train = np.array(embeddings_train)

    #regularize features
    embeddings_train = regularization.regularizeHorizontally(embeddings_train)
	
    return embeddings_train

"""
Trains the linear SVM for subjectivity detection with manually-created feature vectors

inputs:
features_train - list of manually generated feature vectors for each of the messages to be used for training
temp_labels_train - list of labels for subjectivity detection 
    (The labels should be denoted by either 0 for 'neutral' or 1 otherwise)
C - Penalty parameter C of the error term for the SD1/SP1 system

returns:
the trained SVM classifier for SD1

"""
def train_sd1(features_train, temp_labels_train, C):

    sd1 = SVM.train(features_train, temp_labels_train, c=C, k="linear")
    return sd1
	
"""
Trains the linear SVM for polarity classification with manually-created feature vectors

inputs:
features_train - list of feature vectors for each of the messages to be used for training
temp_labels_train - list of labels for subjectivity detection 
    (The labels should be denoted by either 0 for 'neutral' or 1 otherwise)
labels_train - list of labels for polarity classification
    (The labels should be -1 for negative, 0 for neutral, and 1 for positive)
C - Penalty parameter C of the error term for the SD1/SP1 system

returns:
the trained SVM classifier for the SP1
"""
def train_sp1(features_train, temp_labels_train, labels_train, C):
    #calculate the confidence scores for SD1
    train_confidence = calculate_confidence(features_train, temp_labels_train, C)
    #stack the confidence scores into the feature vectors for SP1
    features_train = np.hstack((features_train, train_confidence.reshape(train_confidence.shape[0], 1)))
    #train the SVM classifier for SP1
    sp1 = SVM.train(features_train, labels_train, c=C, k="linear")
    return sp1
	
"""
Trains the linear SVM for subjectivity detection with word embeddings

inputs:
embeddings_train - list of feature vectors built from word embeddings for each of the messages to be used for training
temp_labels_train - list of labels for subjectivity detection 
    (The labels should be denoted by either 0 for 'neutral' or 1 otherwise)
C - Penalty parameter C of the error term for the SD2/SP2 system

returns:
the trained SVM classifier for SD2
"""
def train_sd2(embeddings_train, temp_labels_train, C):
	
    sd2 = SVM.train(embeddings_train, temp_labels_train, c=C, k="linear")
    return sd2

"""
Trains the linear SVM for polarity classification with word embeddings

inputs:
embeddings_train - list of feature vectors built from word embeddings for each of the messages to be used for training
temp_labels_train - list of labels for subjectivity detection 
    (The labels should be denoted by either 0 for 'neutral' or 1 otherwise)
labels_train - list of labels for polarity classification
    (The labels should be -1 for negative, 0 for neutral, and 1 for positive)
C - Penalty parameter C of the error term for the SD2/SP2 system

returns:
the trained SVM classifier for SP2
"""
def train_sp2(embeddings_train, temp_labels_train, labels_train, C):
    #calculate the confidence scores for SD2
    train_confidence = calculate_confidence(embeddings_train, temp_labels_train, C)
    #stack the confidence scores into the feature vectors for SP2
    embeddings_train = np.hstack((embeddings_train, train_confidence.reshape(train_confidence.shape[0], 1)))
    #train the SVM classifier
    sp2 = SVM.train(embeddings_train, labels_train, c=C, k="linear")
    return sp2	
	
"""
Trains the hybrid classifier from the hybrid_classifier folder

inputs: 
tweets - a list of dictionaries containing tweet information.
      Each dictionary should be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
      
returns: 
the trained SVM classifier for the ML classifier in the hybrid classifier pipeline
"""
def train_classifier3(tweets):
    print "Training the hybrid classifier pipeline"
    #create a list of tuples of (message, sentiment) from the input dictionary
    messages = [(tweet['MESSAGE'], tweet['SENTIMENT']) for tweet in tweets]
    #remove the previous trained file, because we want to re-train
    #the TwitterHybridClassifier class will always use the stored file if it exists
    if os.path.exists('model_python2.pkl'):
        os.remove('model_python2.pkl')
    #train the hybrid pipeline classifier
    classifier = TwitterHybridClassifier(messages)
    return classifier
   
"""
Calculates the list of confidence scores to be appended to the feature vectors for
polarity classification

inputs: 
features_train - list of feature vectors for each message to be used for training
labels_train - list of labels for polarity classification
    (The labels should be -1 for negative, 0 for neutral, and 1 for positive)
C - Penalty parameter C of the error term for the system

returns:
A numpy array of normalized confidence scores for the input feature vectors 
and sentiment labels

"""  
def calculate_confidence(features_train, labels_train, C):
    #get confidence score
    train_confidence = getConfidenceScores(features_train, labels_train, C)
	
    #normalize confidence scores
    train_confidence = [softmax(conf) for conf in train_confidence]
	
    train_confidence = np.array(train_confidence)
    return train_confidence

"""
Function used for normalization of the confidence scores.

inputs:
x- an array of confidence scores for a single tweet of the form [<negative>, <neutral>, <positive>]

returns: 
An array of normalized confidence scores for a single tweet of the form [<negative>, <neutral>, <positive>]
"""
def softmax(x):
    return 1 / (1. + math.exp(-x))

"""
Parses the input file names (as arguments), reads all of them, and trains using the combined data
"""
if __name__ == "__main__":
    #checks for invalid input arguments
    if(len(sys.argv) < 3):
        print "Usage : python train.py <# of dimensions for word embeddings vectors> <train_dataset_path>..."
        sys.exit(0)  
    else:
        print str(sys.argv)
        tweets = []
        d = int(sys.argv[1])
        for x in range(2, len(sys.argv)):
            #check if dataset exists
            if os.path.exists(sys.argv[x]):
                print "Loading file: " + sys.argv[x]
                #read all of the tweets from the file and add them to the list 
                #of dictionaries representing the tweets
                tweets += read_file(sys.argv[x])
            else:
                #notify of an error finding the file
                print sys.argv[x] + " could not be found!"
                sys.exit(0)
        print "Training files loaded"
        print "Number of tweets loaded: " + str(len(tweets))
        train(tweets, d)
		
      
