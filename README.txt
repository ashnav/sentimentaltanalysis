Project Roles:
   We spent a fair amount of time performing research on the numerous different ways people have approached performing sentiment analysis on twitter data in the past. All team members participated equally in the research phase. 
   Stephanie wrote all of the code for the project and did all of the testing of the different configurations. Stephanie also downloaded all of the data, and built the arff files for the rest of the team to perform analysis in weka on the feature vectors.
   All team members prepared slides for the presentations.
   All team members collaborated on writing the necessary documentation for the project. Preethi and Ashwini wrote a large portion of the README file( with Stephanie performing some last minute updates and additions to meet the outlined expectations). Xiaolu wrote the majority of the BASELINE file. Xiaolu wrote the original version of the ORIGIN file and Stephanie altered it to create individual files for each of the folders that contain 3rd party source code. Stephanie wrote the INSTALL file. 
   Ashwini and Preethi performed most of the analysis using weka. Xiaolu also did some testing in weka that used all of the tweets rather than a subset.


Author: Preethi Manoharan and Ashwini Murthy, Stephanie Durand

Introduction:

What is Twitter?
Twitter is an online social networking service (microblogging site) that enables people to send and read short 140-character messages called "tweets". Registered users can read and post tweets, but those who are not can only read them. 

Twitter Sentiment Analysis
It is the process of determining the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions and emotions expressed by the people about something. Technically, sentiment analysis is the use of natural language processing, statistics, or machine learning methods to extract, identify, or otherwise characterize the sentiment content of a text unit.

Objective
To determine whether the expressed opinion in the tweet is either positive, negative or neutral.

Problem
Generally, the tweet texts are 
Short: most of the cases, a sentence or a headline rather than a document.  
The language used is informal. 
Highly unstructured and non-grammatical tweets
Words with no Vocabulary
Lexical variations
Extensive usage of acronyms like asap, lol, afaik

———————————————————————————————————————————————————

Program Input: 
A file of tweets where each line has following format:
Format: <tweet id><tab><sentiment label><tab><tweet content>.
Actual Input: 629226490152914944	Positive		Microsoft, I may not prefer your gaming branch of business. But, you do make a damn fine operating system. #Windows10 @Microsoft

Program Output: 
A file of tweet ids to sentiment label in the following format:
Format: <tweet id><tab><predicted sentiment label>
Actual Output: 629226490152914944	Positive

Example input files can be seen within the data folder. 
For training, data/train60
For testing, data/test40 (these are currently the same format)

Example output file can also be seen within the data folder:
data/combined_results

———————————————————————————————————————————————————

Methods:

We used the weighted ensemble of SVM (Support Vector Machine) as one of our baseline methods. This system participated in the SemEval 2016 Task 4 event last year and placed 5th out of 34. The paper and code for this approach can be found at https://aclweb.org/anthology/S/S16/S16-1012.pdf and https://github.com/nlpaueb/aueb.twitter.sentiment, respectively. Our overall method uses this initial implementation and improves it by adding an additional classifier into the ensemble. 
As evidenced by its title, this method is based on an ensemble of SVMs. A Support Vector Machine (SVM) is a classifier which can be defined by a separating hyperplane. When labeled training data is given, the algorithm outputs an optimal hyperplane which categorizes new examples. This is a form of supervised learning.
Before being provided to the classifiers, each tweet is preprocessed, which includes tokenization and POS (Part-of-Speech)tagging. For tokenization, this method used a tokenizer specifically tailored for tweets called twokenize. This particular tokenizer is a crude Python port of the Twokenize class from the ark-tweet-nlp toolkit. It can easily replace tweet features like usernames, urls, phone numbers, times, etc. with tokens in order to reduce feature set complexity and improve the performance of classifiers. For part-of-speech tagging, this method also used the ark-tweet-nlp tagger, a Java-based POS tagger from Carnegie Mellon. 
After preprocessing, the tweets are then provided to the system of classifiers for sentiment detection. In the weighted ensemble of SVMs method, we have two sets of classifiers.  SD1 and SP1 form  system I and SD2 & SP2 form system II. Basically, we have two classifiers per system, SD1/2 which are used for subjectivity detection (whether a tweet is neutral or contains some emotion) and SP1/2 which are used for sentiment polarity classification (for detecting the polarity of the sentiment (Positive or Negative)). System I (SD1 and SP1) uses a manually-generated feature vector for training and testing based on morphology, parts-of-speech(POS), a variety of sentiment lexicons, negations, word clusters, and the presence of slang. A complete list of these features can be seen at the end of this README file. SP1 then uses the same feature vector as SD1, but adds the confidence score of SD1 as additional features. The confidence scores of an SVM are based on the distance from the separating hyperplane. The farther the item was from the separating hyperplane, the more confident we are that it meets that classification.

System II is based on word embeddings. SD2 (subjectivity detection classifier) and SP2
(polarity classifier) form the system II. This system uses pre-trained word embeddings from the Stanford GloVe (Global Vector) project that were trained on approximately 2 billion tweets and 27 billion tokens. The centroid of all of the word vectors (determined from the pre-trained word embeddings) of each tweet is considered as feature vector and it is given to SD2 and SP2.

We used 100 dimensional word vectors for twitter produced by GloVe(Global Vectors) instead of 200 dimensional vectors(originally used by the paper) since  there were challenges in running  200 dimensional vectors (memory constraint). Like system I, the confidence scores  of SD2 were added to the feature vector for SD2 to build the feature vector for SP2. This system has been improved from strict 2-stage approach by considering the confidence scores from the subjectivity detection classifiers. 

To improve the weighted ensemble method, we added an additional classifier, NILC_USP.  The paper and code for this approach can be found at http://www.pedrobalage.com/pubs/BalageFilho2014NILCUSPImprovedHybrid.pdf and https://github.com/pedrobalage/TwitterHybridClassifier, respectively.

Normalization 
Tweet messages are normalised using transfer hashtags, urls, and classified emoticons which are converted into codes and part-of-speech tagging is performed using ark-tweet-nlp.

Rule-based classification
The actual polarity of the tweets is indicated by the presence of emoticons. Example: If the number of negative emoticons is greater than the positive emoticons then the system outputs the tweet as “Negative”.

Lexicon-based classification
Each word in the lexicon is compared to sentiment lexicons and classifies as a positive tweet as if the score is +1, negative tweet if the score is -1 and neutral tweet if the score is 0. 

Machine learning classification:
Linear Kernel SVM classifier (similar to that used in SD1/SP1) using the following features 
unigrams, bigrams and trigrams  
Presence of negation  
Presence of three or more characters in the words 
Sequence of three or more punctuation marks 
Number of words with all letters in uppercase  
Total number of each tag present in the text 
Number of positive words computed by the lexicon-based method  
Number of negative words computed by the lexicon-based method.

The correlation between the rule-based classifier scores and the gold standard classes in the DevSet, as well as between the lexicon-based classifier scores and the gold standard classes. These were used to select the scores used to determine whether a certain stage of the pipeline made the classification decision. If the rules-based classifier returned anything <= -1 or => 1, this classifier would make the prediction. Otherwise, the pipeline would continue on to the lexicon-based classifier. Likewise, it the lexicon-based classifier returned a score <= - 2 or >= 1, it would make the decision, otherwise the tweet would continue to the machine learning (SVM) classifier. The machine learning classifier has a bias, so if the negative confidence score of the SVM classifier was greater than -0.4, the tweet is classified was Positive. Otherwise, the tweet was classified by whichever confidence score was higher. 

———————————————————————————————————————————————————

Improved Method

The weighted ensemble of SVMs method is extended to include an additional pipeline classifier which is a combination of rules-based, lexicon-based and machine learning classifiers. We decided to combine these two approaches because the pipeline method noticed an increase in performance from including the more rudimentary rules-based and lexicon-based classifiers.
The entire ensemble is of the form:

C  = W1* C(SP1) + W2*C(SP2) + W3*C(PIPE)

When the pipeline returns a result from the rules based or lexicon based classifier, we set the the confidence score for the returned sentiment to 1 and the others to 0. Example: lexicon-based classifies message as negative, so C(PIPE) = [1,0,0] i.e [Negative, Neutral, Positive]. Good choices of weights were selected by randomizing them and then choosing the set that maximized the performance. These weights were chosen by saving off the confidence scores for each of the classifiers and running 10000 iterations of using a random number generator to select the weights such that W1+W2+W3=1. This was generated from running our classifiers on a 90/10 split of the data. Ultimately, we found that the classifier performed better overall with a slightly altered version of the weights used in the original weighted SVM classifier. We used weights W1=0.58, W2=0.34, W3=0.08 to obtain the best overall accuracy. 

We used a combination of all of the training and test data available from SemEval for this task. This included the training and dev data from 2013 and all of the training, dev, and devtest data from 2016. This accumulated to a total of 17118 tweets used to perform 10-fold cross validation. 

After our results from the inital phase of the project, we observed that the classifier was performing poorly on negative and neutral tweets. In order to improve performance on these types of tweets, we first added in the additional training data (increased from ~5000 tweets for our initial testing to 17118 tweets for the second phase) and then performed analysis using weka, an open source data mining tool, to analyze the relative information gain for the manually-generated feature vectors of the SD1/SP1 system and the ML classifier from the pipeline to determine if any of these features were potentially causing a degradation of our performance. We also used WEKA to evaluate the use of a different type of features as well as different types of classifiers. From this analysis, we determined that we could remove over 1000 of the 1316 features in the original feature vector used within the SD1/SP1 system. We selected only the features that had information gain > 0.05. This reduced the number of features from 1316 to 120. The final list of features is listed below. We also decided to keep all of the features from the pipeline ML classifier because they all had fairly similar information gain. Additional information on the analysis performed using weka is included in the appendix of this file. We saw good performace with a relatively small amount of data using the StringToWordVector filter within weka, a term frequency-inverse document frequenct metric, but we were unable to replicate the results in our own code with independent 10-fold cross validation. We decided not to include a classifier that used this representation of the tweets because of its poor performance on the larger set of data. 

To verify that the removal of features did not degrade our performance, we ran 10-fold cross validation with our large dataset on the original weighted SVM classifier with the pruned feature set. We did not see a large degradation of performance, and decided that removing the features was in our benefit. The features of the pipeline's ML classifier all had very little information gain and we had many features in the SD1/SP1 system based on lexicons, so we attempted eliminating the lexicon-based classifier and ML classifier from the pipeline and building a "mini-pipeline" that used the rules-based classifier as a first step and if that classifier does not obtain a certain level of confidence on a tweet, the weighted ensemble of SVMs with the pruned feature set would be used to classify the tweet. When we tested this configuration using 10-fold cross-validation, we found that it performed better than just the weighted ensemble of SVMs with the pruned feature set, but not quite as well as the weighted ensemble of the SD1/SP1, SD2/SP2, and the whole pipeline classifier. 

We found that the  best configuration for our approach was to add the pipeline classifier as a third member of the weighted ensemble, with the reduced number of features in the SD1/SP1 system.


———————————————————————————————————————————————————

Final set of Features used in the SD1/SP1 system 

Morphological features
-The existence of date references
-The number of tokens that start with an upper case letter
-The number of exclamation marks
-The sum of exclamation and question marks
-The number of tokens containing only exclamation marks
-The number of tokens containing only exclamation or question marks
-The existence of a subjective (i.e., positive or negative) emoticon at the message’s end
-The existence of an ellipsis and a link at the message’s end
-The existence of an exclamation mark at the message’s end

POS based features
-The number of adjectives
-The number of adverbs
-The number of interjections
-The number of verbs
-The number of proper nouns
-The average, maximum F1 scores of the message’s POS unigrams for the negative class
-The average, maximum F1 scores of the message’s POS bigram for the negative class

Sentiment lexicon based features
-Sum of scores
-Maximum of scores
-Average of scores
-The count of words with scores
-The score of the last word of the message that appears in the lexicon
-The score of the last word of the message

Miscellaneous features
-Negation


———————————————————————————————————————————————————

Appendix: Additional details on weka analysis

As part of phase II of the project we have used weka – an open source data mining tool in our project. Weka contains tools for data pre-processing, classification, regression, clustering, association rules, and visualization. 


Summary
First we started by importing the tweet dataset in WEKA, then we experimented some text preprocessing tasks such as word extraction, stop-words removal, stemming and term selection. Finally, we ran various classification algorithms (naive bayes, SMO) and compared the results, in terms of classification accuracy.


A relatively small dataset of about 8000 tweets were taken. Each tweet is converted to .txt files with the help of R programming which is then placed in a directory. Then under directory we are placing two sub-directories with names being POS and NEG. These subdirectories serve as class labels for the “Text directory loader” of weka. As the output of the R Program goes to the standard output, each tweet should be redirected it to a .txt file. We have left the output of the execution of this command for both the POS and the NEG collections of .txt files into a separate folder.


For the classification task to be done, a preliminary phase of text preprocessing and feature extraction is essential. We wanted to transform each text in a vector form, in which each document is represented by the presence (or frequency) of some “important” terms; this terms are the ones contained in the collection vocabulary. To build the vocabulary, various operations are typically performed (many of which are language-specific):


Word parsing and tokenization
In this phase, each document is analyzed with the purpose of extracting the terms. Separator characters must be defined, along with a tokenization strategy for particular cases such as accented words, hyphenated words, acronyms, etc.
Stop-words removal
A very common technique is the elimination of frequent usage words: conjunctions, prepositions, base verbs, etc. This kind of terms should be filtered as they have a poor characterizing power, making them useless for the text classification.
Lemmatization and stemming
The lemmatization of a word is the process of determining its lemma. The lemma can be thought of as the “common root” of various related inflectional forms; for instance, the words walk, walking and walked all derive from the lemma walk.


A simple technique for approximated lemmatization is the stemming. Stemming algorithms work by removing the suffix of the word, according to some grammatical rules.
Term selection/feature extraction
The term set resulting from the previous phases has still to be filtered, since we need to remove the terms that have poor prediction ability (w.r.t the document class) or are strongly correlated to other terms. This term selection task also leads to a simpler and more efficient classification.
To perform the preprocessing in WEKA, we used the StringToWordVector filter from the package weka.filters.unsupervised.attribute. 
Configured the tokenizer (term separators);
Specified a stop-words list;
Chose a stemmer (optional).
The preprocessing operations will affect the quality of the classification, for this reason we will perform various experiments on different generated datasets.
The default text retrieval model used by the StringToWordVector filter is boolean: each document is represented with an n-dimensional boolean vector, where n is the size of the vocabulary, and each value models the presence or the absence of a vocabulary term in the document. 


In order to construct an ARFF file from this structure, we use the weka.core.converters.TextDirectoryLoader class, which is an evolution of a previously existing helper class named TextDirectoryToArff.java and available at WEKA Documentation at wikispaces. Using this class is as simple as issuing the next command:
$> java weka.core.converters.TextDirectoryLoader -dir NLP_WEKA > POS
$> java weka.core.converters.TextDirectoryLoader -dir NLP_WEKA > NEG


You have to call this command at the parent directory of NLP_WEKA, and the parameter -dir sets up the input directory.


Keeping in mind that pipelined features is the problem, the relevant features needed to be selected using Information Gain. Used to calculate information gain of each feature for subjectivity detection and sentiment polarity classifiers and rank the attributes based on the information gain. Eliminated the features based on the ranking and calculated the classification accuracy.


As a first step in weka we derive the following calls to the weka.filters.unsupervised.attribute.StringToWordVector class:


$> java weka.filters.unsupervised.attribute.StringToWordVector -R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler "weka.core.stopwords.WordsFromFile -stopwords /home/ashwini/Desktop/NLP/Stopwordslist.txt" -M 1 -tokenizer "weka.core.tokenizers.WordTokenizer -delimiters \" \\r\\n\\t.,;:\\\'\\\"()?!\"" 
To perform the 10 fold cross validation, we have to chain the filters (StringToWordVector and AttributeSelection) and the classifier in order to perform a valid experiment, as the features for each folder should be different. We are using the following classification algorithms:


Tweet Classification Algorithms:
Naive Bayes Algorithm
SVM based SMO classifier


We got the following results when we worked on WEKA:
Naive Bayes Classifier with Stopwords list
1.  Percentage Split as 60:40


Correctly Classified Instances        2167               69.2111 %
Incorrectly Classified Instances       964               30.7889 %


2. 10 Cross Validation


Correctly Classified Instances        5640               72.0583 %
Incorrectly Classified Instances      2187               27.9417 %




Naive Bayes Classifier with Stopwords list and Stemmer 
1. Percentage Split as 60:40
Correctly Classified Instances        2167               69.2111 %
Incorrectly Classified Instances       964               30.7889 %


2. 10 Cross Validation
Correctly Classified Instances        5640               72.0583 %
Incorrectly Classified Instances      2187               27.9417 %




 SMO with Stopwords list and Stemmer
1. Percentage Split as 60:40
Correctly Classified Instances        2508               80.1022 %
Incorrectly Classified Instances       623               19.8978 %


2. 10 Cross Validation
Correctly Classified Instances        6367               81.3466 %
Incorrectly Classified Instances      1460               18.6534 %


 SMO with Stopwords list  
1. Percentage Split as 60:40
Correctly Classified Instances        2508               80.1022 %
Incorrectly Classified Instances       623               19.8978 %


2. 10 Cross Validation
Correctly Classified Instances        6367               81.3466 %
Incorrectly Classified Instances      1460               18.6534 %


SMO with Stopwords list  10 fold cross validation (train 60 + test 40 from github)




=== Run information ===


Scheme:       weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"


Relation:     C__Users_windows_Desktop_Sentiments-weka.filters.unsupervised.attribute.StringToWordVector-R1-W1000-prune-rate-1.0-N0-stemmerweka.core.stemmers.NullStemmer-stopwords-handlerweka.core.stopwords.WordsFromFile -stopwords C:\Users\windows\Desktop\Stopwordslist.txt-M1-tokenizerweka.core.tokenizers.WordTokenizer -delimiters " \r \t.,;:\'\"()?!"
Instances:    2536
Attributes:   4752
              [list of attributes omitted]
Test mode:    10-fold cross-validation


=== Classifier model (full training set) ===


SMO
Kernel used:
 Linear Kernel: K(x,y) = <x,y>
Classifier for classes: NEG, POS


BinarySMO




=== Stratified cross-validation ===
=== Summary ===


Correctly Classified Instances        1620               63.8801 %
Incorrectly Classified Instances       916               36.1199 %
Kappa statistic                         -0.0221
Mean absolute error                      0.3612
Root mean squared error                  0.601 
Relative absolute error                 90.3542 %
Root relative squared error            134.4426 %
Total Number of Instances             2536     




=== Confusion Matrix ===


    a    b   <-- classified as
  111  589 |    a = NEG
  327 1509 |    b = POS


=== Run information ===
NaïveBayes with Stopwords list  10 fold cross validation (train 60 + test 40 from github)


Scheme:       weka.classifiers.bayes.NaiveBayes 
Relation:     C__Users_windows_Desktop_Sentiments-weka.filters.unsupervised.attribute.StringToWordVector-R1-W1000-prune-rate-1.0-N0-stemmerweka.core.stemmers.NullStemmer-stopwords-handlerweka.core.stopwords.WordsFromFile -stopwords C:\Users\windows\Desktop\Stopwordslist.txt-M1-tokenizerweka.core.tokenizers.WordTokenizer -delimiters " \r \t.,;:\'\"()?!"
Instances:    2536
Attributes:   4752
              [list of attributes omitted]
Test mode:    10-fold cross-validation


=== Classifier model (full training set) ===


Naive Bayes Classifier
=== Stratified cross-validation ===
=== Summary ===


Correctly Classified Instances        1879               74.0931 %
Incorrectly Classified Instances       657               25.9069 %
Kappa statistic                          0.332 
Mean absolute error                      0.3025
Root mean squared error                  0.4377
Relative absolute error                 75.6614 %
Root relative squared error             97.9119 %
Total Number of Instances             2536      


=== Confusion Matrix ===
    a    b   <-- classified as
  338  362 |    a = NEG
  295 1541 |    b = POS
=== Run information ===
SMO with Stopwords list  Split 60:40 (train 60 + test 40 from github)




Scheme:       weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
Relation:     C__Users_windows_Desktop_Sentiments-weka.filters.unsupervised.attribute.StringToWordVector-R1-W1000-prune-rate-1.0-N0-stemmerweka.core.stemmers.NullStemmer-stopwords-handlerweka.core.stopwords.WordsFromFile -stopwords C:\Users\windows\Desktop\Stopwordslist.txt-M1-tokenizerweka.core.tokenizers.WordTokenizer -delimiters " \r \t.,;:\'\"()?!"
Instances:    2536
Attributes:   4752
              [list of attributes omitted]
Test mode:    split 60.0% train, remainder test


=== Classifier model (full training set) ===


SMO
Kernel used:
 Linear Kernel: K(x,y) = <x,y>
Classifier for classes: NEG, POS


BinarySMO
=== Summary ===


Correctly Classified Instances         696               68.6391 %
Incorrectly Classified Instances       318               31.3609 %
Kappa statistic                          0.0789
Mean absolute error                      0.3136
Root mean squared error                  0.56  
Relative absolute error                 78.6539 %
Root relative squared error            126.3432 %
Total Number of Instances             1014     


=== Confusion Matrix ===
   a   b   <-- classified as
  56 216 |   a = NEG
 102 640 |   b = POS

