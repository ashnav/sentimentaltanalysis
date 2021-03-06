# 
# This file is not used on our final classifier. This builds a 
# classifier using the StringToWrodVector filter and SMO classifier in weka  via the 
# python weka wrapper. The results we found were much worse than we had anticipated,
# so we opted to not use this classifier within our weighted combination of 
# classifiers.
#
#
# Author: Stephanie Durand
#
from weka.core.dataset import *
from weka.classifiers import Classifier
from weka.filters import StringToWordVector
from weka.core.stemmers import Stemmer
from weka.core.stopwords import Stopwords
from weka.core.tokenizers import Tokenizer
from weka.core.converters import Loader
import tempfile
import os

class StringWordVectorClassifier:
    
    """
    Constructor for the StringWordVectorClassifier. 
    This will perform the training step with the provided list of tweets
    
    inputs:
    training_data - a list of tweet dictionaries
          Each dictionary should be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
    
    """
    def __init__(self, training):
        #build data in the format weka expects
        dataset = StringWordVectorClassifier.convertDictToInstances(training)
        
        #build filter
        stemmer = Stemmer(classname='weka.core.stemmers.NullStemmer')
        fileDir = os.path.dirname(__file__)
        stopwordsfile = os.path.join(fileDir, 'resources/Stopwordslist.txt')
        stopwords = Stopwords(classname='weka.core.stopwords.WordsFromFile', options=['-stopwords', stopwordsfile])
        tokenizer = Tokenizer(classname='weka.core.tokenizers.WordTokenizer', options=['-delimiters', "\" \r\n\t.,;:\'()?!"])
        self.str2wv = StringToWordVector(options=['-R', 'first-last', '-W', '1000', '-prune-rate','-1.0', '-N','0'])
        self.str2wv.stemmer = stemmer
        self.str2wv.stopwords = stopwords
        self.str2wv.tokenizer = tokenizer
        self.str2wv.inputformat(dataset)
        print "Building Word Vectors"
        filtered = self.str2wv.filter(dataset)
        
        #build classifier
        print "Building SMO CLassifier"
        self.cls = Classifier(classname="weka.classifiers.functions.SMO")
        self.cls.build_classifier(filtered)
        
    """
    Classifies the provided list of tweets with the pre-trained SMO classifier.
    inputs: 
    testing -  a list of tweet dictionaries
          Each dictionary should be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
    
    outputs:
    a list containing only the predicted sentiment of each tweet
    """    
    def classify(self, testing):
        print 'Performing classification'
        test_set = StringWordVectorClassifier.convertDictToInstances(testing)
        filtered_test = self.str2wv.filter(test_set)
        predictions = []
        for inst in filtered_test:
            predNum = self.cls.classify_instance(inst)
            predictions.append(filtered_test.class_attribute.value(int(predNum)))
        return predictions
    
    """
    Converts from a python dictionary of tweets to the weka Instances data format 
    for use within the weka filters and classifiers.
    inputs:
     a list of tweet dictionaries
          Each dictionary should be of the form:
      {
         'ID': <id>,
         'SENTIMENT': <sentiment>,
         'MESSAGE': <tweet message>
      }
      
     outputs:
     an Instances object containing all of the tweet data in the format used by weka
    """
    @staticmethod    
    def convertDictToInstances(tweets):
        header_string = "@relation tweets_sentiment \n@attribute \'tweet\'\tstring\n@attribute \'sentiment\'\t{positive,neutral,negative}\n@data\n"
        temp_ptr, temp_file = tempfile.mkstemp(suffix='.arff')
        
        with open(temp_file, "w") as arff:
            arff.write(header_string)
            for x in range(0, len(tweets)):
                data = "\'{tweet}\',{sentiment}\n".format(sentiment=tweets[x]['SENTIMENT'],tweet=tweets[x]['MESSAGE'].strip().replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"'))
                arff.write(data)
        os.close(temp_ptr)
        
        loader=Loader(classname='weka.core.converters.ArffLoader')
        data=loader.load_file(temp_file)
        data.class_is_last()
        os.remove(temp_file)
        return data
        
"""
Method used to test classifier. The data file that it uses is hard-coded in because it 
is strictly for testing purposes.
"""        
if __name__ == "__main__":
    import weka.core.jvm as jvm
    import sys
    sys.path.append("../")
    from twitter_data_reader import read_file
    tweets = read_file("../../data/semeval_data.tsv")
    print len(tweets)
    from cross_validate import fold_data
    folds = fold_data(tweets, 10)
    try:
        jvm.start()
        for foldNum in range(0,len(folds)):
            print "Starting fold {}".format(foldNum)
            training = []
            test = []
            for index in range(0, len(folds)):
                if index == foldNum:
                    test = folds[index]
                else:
                    training.extend(folds[index])
                    
            classifier = StringWordVectorClassifier(training)
            predictions = classifier.classify(test)
            with open('wv_fold' + str(foldNum), 'w') as outfile:
                for i in range(0,len(test)):
                    outfile.write('{}\t{}\n'.format(tweets[i]['ID'], predictions[i]))
        
    finally:
        jvm.stop()