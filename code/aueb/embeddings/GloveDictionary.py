import numpy as np
import sys
#sys.path.append('../features')
#from features import morphologicalFeatures,posBasedFeatures
from features.morphologicalFeatures import *
from features.posBasedFeatures import *

#Glove word embeddings
class Glove() :

    #file directory
    directory = "embeddings/Glove/"
    
    #constructor
    def __init__(self,d=25):	
        #word list
        self.embeddings = {}

	#select embeddings size
        if d == 25:
            self.size=25
            self.file = "glove.twitter.27B.25d.txt"
        elif d == 50:
            self.size=50
            self.file = "glove.twitter.27B.50d.txt"
        elif d == 100:
            self.size=100
            self.file = "glove.twitter.27B.100d.txt"
        elif d == 200:
            self.size=200
            self.file = "glove.twitter.27B.200d.txt"
        else:
            print "Invalid embeddings size"
        
        #load embeddings
        self.loadEmbeddings()
        
		

    #load Slang Dictionary
    def loadEmbeddings(self):
        with open(Glove.directory+self.file, 'r') as fIn:
            #for line in f.readlines():
            for line in fIn:
                
                line = line.decode('utf8')
                #remove \n characters
                line = line.rstrip()
                line_split = line.split(' ')
                word = line_split[0]
                emb = [float(x) for x in line_split[1:]]                
                #add new data
                self.embeddings[word] = np.array(emb)

	#find the embeddings for a given word
    def findWordEmbeddings(self,word):
        return self.embeddings.get(word, None)

	#find the centroid embedding for a given message
    def findCentroid(self,tokens):
        counter=0
        #initialize centroid
        centroid = [0 for x in range(0,self.size)]
        for token in tokens:
            #find embeddings for token
            emb = self.findWordEmbeddings(token)
	    #add only non 0 embeddings to centroid	
            if emb is not None:   
                centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
                counter+=1
        #divide with size
        try:
            centroid = [x/float(counter) for x in centroid]
        except:
            pass
        
        return centroid
