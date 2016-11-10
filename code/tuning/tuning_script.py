#
# This is a script for tuning the combination weights to achieve the highest
# F-1 measure for the ensemble of classifiers.
# 
# Authors: Stephanie Durand
# 

import numpy as np

"""
Reads the confidence scores files
inputs:
f - the file path for the confidence scores of one classifier

returns:
a 2-D array of confidence scores where each row is of the form:
[negative confidence score, neutral confidence score, positive confidence score]
and there is a row for each message.
"""
def read_confidence_scores(f):
    confidence_scores = []
    with open(f, 'r') as scores:
        for line in scores:
            split_line = line.strip().split("\t")
            message_conf =[]
            for x in range(0,3):
                sentiment_conf = float(split_line[x].strip())
                message_conf.append(sentiment_conf)
            message_conf = np.array(message_conf)
            confidence_scores.append(message_conf)
        confidence_scores = np.array(confidence_scores)
        return confidence_scores
    
"""
Predicts the sentiment labels using the confidence scores and 
weights provided. 

inputs:
confidence_sp1 - a 2-D array of confidence scores for the SP1 classifier
confidence_sp2 - a 2-D array of confidence scores for the SP2 classifier
confidence_pipe - a 2-D array of confidence scores for the pipeline classifier
W1 - the weight to apply to the confidence scores for SP1
W2 - the weight to apply to the confidence scores for SP2
W3 - the weight to apply to the condifence scores for the pipeline classifier

returns:
the list of sentiment predictions as strings (either 'positive', 'negative', or 'neutral')
"""    
def predict(confidence_sp1,W1, confidence_sp2, W2, confidence_pipe, W3):
    
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
        results.append(pol)
    return results

"""
Randomizes the weight to use when combining the confidence scores and 
calculates the sentiment label predictions for the ensemble.

inputs:
confidence_sp1 - a 2-D array of confidence scores for the SP1 classifier
confidence_sp2 - a 2-D array of confidence scores for the SP2 classifier
confidence_pipe - a 2-D array of confidence scores for the pipeline classifier

outputs:
a tuple containing the following data at the noted index
 0 - the list of sentiment predictions as strings (either 'positive', 'negative', or 'neutral')
    from using the random weights
 1 - the random weight for SP1, 
 2 - the random weight for SP2, 
 3 - the random weight for the hybrid classifier pipeline

"""
def randomize_weights(confidence_sp1, confidence_sp2, confidence_pipe):
    W1 = np.random.uniform(0,1)
    W2 = np.random.uniform(0,1-W1)
    W3 = 1 - (W1+W2)
    
    predictions = predict(confidence_sp1, W1, confidence_sp2, W2, confidence_pipe, W3)
    return (W1, W2, W3, predictions)
    
"""
Calculates the F1 score using the same formula as the SemEval organizers. (I looked at the 
scoring script for subtask A and wrote the F1 calculation in Python.) The input 
true_labels and predictions list must be in the same order. This means that index 
x in the list is the sentiment label for tweet x. 

inputs:
true_labels - a list of the true labels for the tweets
predictions - a list of the predicted labels for the tweets

returns: 
the overall F1 measure (as a float)
"""    
def calculate_F1(true_labels, predictions):
    #build the data storage structure
    stats = {}
    classes = ['negative', 'neutral', 'positive']
    for clazz in classes:
        sub_dict = {}
        for clazz2 in classes:
            sub_dict[clazz2] = 0
        stats[clazz] = sub_dict
    
    #fill in the data structure
    length = min(len(true_labels), len(predictions))
    for x in range(0, length):
        stats[predictions[x]][true_labels[x]] += 1
    
    #calculate F1
    overall = 0.0
    
    for clazz in classes:
        possible_denomP = stats['positive'][clazz] + stats['negative'][clazz] + stats['neutral'][clazz]
        denomP =  possible_denomP if possible_denomP > 0 else 1
        P = 100.0 * stats[clazz][clazz] / denomP
        
        possible_denomR = stats[clazz]['positive'] + stats[clazz]['negative'] + stats[clazz]['neutral']
        denomR = possible_denomR if possible_denomR > 0 else 1
        R = 100.0 * stats[clazz][clazz] / denomR
        
        denom = (P + R) if (P + R) > 0 else 1
        F1 = 2*P*R / denom
        if clazz != 'neutral':
            overall += F1
    overall = overall/2.0
    return overall
    
"""
The main functionality of the script. 

This will read in all of the confidence score files as well as the 
file containing the true labels for the tweets. 
It will then randomize the weights used for combining the classifiers 100,000 times 
and print out the set of weights that produce the highest overall F1 measure.
"""    
if __name__=="__main__":
    import sys
    sys.path.append("../")
    if(len(sys.argv) < 5):
        print "Usage : python tuning_script.py <file with correct tweet labels> <sp1 confidence scores file> <sp2 confidence scores file> <pipeline confidence scores file>"
    print "Reading the truth file"
    truth_file = sys.argv[1]
    from twitter_data_reader import read_file
    tweets = read_file(truth_file)
    true_labels = [d['SENTIMENT'] for d in tweets]
    
    print "Reading the confidence score files"
    file_sp1 = sys.argv[2]
    file_sp2 = sys.argv[3]
    file_pipe = sys.argv[4]
    confidence_sp1 = read_confidence_scores(file_sp1)
    confidence_sp2 = read_confidence_scores(file_sp2)
    confidence_pipe = read_confidence_scores(file_pipe)
    maxF1 = 0.0
    best_weights = [0.0, 0.0, 0.0]
    
    print "Generating random weights and testing F1 measure"
    for x in range(0,100000):
        if x % 10000 == 0:
            print "Running iteration " + str(x)
        results = randomize_weights(confidence_sp1, confidence_sp2, confidence_pipe)
        predictions = results[len(results)-1]
        F1 = calculate_F1(true_labels, predictions)
        if F1 > maxF1:
            maxF1 = F1
            for x in range(0,3):
                best_weights[x] = results[x]
    
    print "Best overall F1:" + str(maxF1)
    print "Best weights: " + str(best_weights)
    
    
    
    
