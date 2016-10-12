import numpy as np

#setup the file paths
file_sp1 = "confidence_sp1"
file_sp2 = "confidence_sp2"
file_pipe = "confidence_pipe"
file_input_messages = "../../data/test40"

"""

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
        
def randomize_weights(confidence_sp1, confidence_sp2, confidence_pipe):
    W1 = np.random.uniform(0,1)
    W2 = np.random.uniform(0,1-W1)
    W3 = 1 - (W1+W2)
    
    predictions = predict(confidence_sp1, W1, confidence_sp2, W2, confidence_pipe, W3)
    return (W1, W2, W3, predictions)
    
    
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
    
    
if __name__=="__main__":
    import sys
    sys.path.append("../")
    from twitter_data_reader import read_file
    tweets = read_file(file_input_messages)
    true_labels = [d['SENTIMENT'] for d in tweets]
    
    confidence_sp1 = read_confidence_scores(file_sp1)
    confidence_sp2 = read_confidence_scores(file_sp2)
    confidence_pipe = read_confidence_scores(file_pipe)
    maxF1 = 0.0
    best_weights = [0.0, 0.0, 0.0]
    
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
    
    print maxF1
    print best_weights
    
    
    
    