Project Roles:
   We spent a fair amount of time performing research on the numerous different ways people have approached performing sentiment analysis on twitter data in the past. All team members participated equally in the research phase. 
   Stephanie wrote all of the code for the project, but there was not a lot of it because we are mostly combining two approaches for which we already had access to the code. 
   Preethi, Ashwini, and Xiaolu prepared the slides for the presentation.
   All team members collaborated on writing the necessary documentation for the project. Preethi and Ashwini wrote the majority of the README file, Xiaolu wrote the majority of the BASELINE file. Xiaolu wrote the original version of the ORIGIN file and Stephanie altered it to create individual files for each of the folders that contain 3rd party source code. Stephanie wrote the INSTALL file. 

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
After preprocessing, the tweets are then provided to the system of classifiers for sentiment detection. In the weighted ensemble of SVMs method, we have two sets of classifiers.  SD1 and SP1 form  system I and SD2 & SP2 forms system II. Basically, we have two classifiers per system, SD1/2 which are used for subjectivity detection (whether a tweet is neutral or contains some emotion) and SP1/2 which are used for sentiment polarity classification (for detecting the polarity of the sentiment (Positive or Negative)). System I (SD1 and SP1) uses a manually-generated feature vector for training and testing based on morphology, parts-of-speech(POS), a variety of sentiment lexicons, negations, word clusters, and the presence of slang. A complete list of these features can be seen at the end of this README file. SP1 then uses the same feature vector as SD1, but adds the confidence score of SD1 as additional features. The confidence scores of an SVM are based on the distance from the separating hyperplane. The farther the item was from the separating hyperplane, the more confident we are that it meets that classification.

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

When the pipeline returns a result from the rules based or lexicon based classifier, we set the the confidence score for the returned sentiment to 1 and the others to 0. Example: lexicon-based classifies message as negative, so C(PIPE) = [1,0,0] i.e [Negative, Neutral, Positive]. Good choices of weights were selected by randomizing them and then choosing the set that maximized the performance. 
W1 = 0.1, W2 = 0.75, W3 = 0.15

———————————————————————————————————————————————————

Features used in the SD1/SP1 system 
Morphological features
The existence of elongated tokens (e.g., “baaad”)
The number of elongated tokens
The existence of date references
The existence of time references
The number of tokens that contain only upper case letters
The number of tokens that contain both upper and lower case letters
The number of tokens that start with an upper case letter
The number of exclamation marks
The number of question marks
The sum of exclamation and question marks
The number of tokens containing only exclamation marks
The number of tokens containing only question marks
The number of tokens containing only exclamation or question marks
The number of tokens containing only ellipsis(...)
The existence of a subjective (i.e., positive or negative) emoticon at the message’s end
The existence of an ellipsis and a link at the message’s end
The existence of an exclamation mark at the message’s end
The existence of a question mark at the message’s end
The existence of a question or an exclamation mark at the message’s end
The existence of slang
POS based features
The number of adjectives
The number of adverbs
The number of interjections
The number of verbs
The number of nouns
The number of proper nouns
The number of urls
The number of subjective emoticons
The average, maximum and minimum F1 scores of the message’s POS unigrams for the positive , negative and the neutral class
The average, maximum and minimum F1 scores of the message’s POS bigram for the positive , negative and the neutral class
The average, maximum and minimum F1 scores of the message’s POS trigram for the positive , negative and the neutral class
Sentiment lexicon based features
Sum of scores
Maximum of scores
Minimum of scores
Average of scores
The count of words with scores
The score of the last word of the message that appears in the lexicon
The score of the last word of the message
Miscellaneous features
Negation
Carnegie Mellon University’s Twitter clusters(938 features)

———————————————————————————————————————————————————

Author: Preethi Manoharan and Ashwini Murthy
