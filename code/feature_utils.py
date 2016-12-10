# 
# As our project was based mostly on combining techniques with existing code, there was not a lot of code, so 
# Stephanie Durand wrote or adapted all of the code for the project phase I and II.
#
# This code is based off of the file features.py from the aueb project folder.
# We created this adaptation so that we could specify the features that we want 
# to include from a file and only include the specified features in the
# manually created feature vector for the sd1/sp1 classifiers of the 
# weighted ensemble of SVMs. This is how we were able to reduce the number of 
# features in the feature vector.
#  
#
# Adapted from code written by:
# Stavros Giorgis, Apostolos Rousas, John Pavlopoulos, 
# Prodromos Malakasiotis and Ion Androutsopoulos
#
#
# Adapted by: Stephanie Durand
#
import numpy as np
from aueb.features.morphologicalFeatures import *
from aueb.features.posBasedFeatures import *
from aueb.features.lexiconBasedFeatures import *
from aueb.features.posBigramsFeatures import *
from aueb.features.otherFeatures import *
from aueb.features.clusterFeatures import *

"""
return feautures of a list of messages as an array
"""
def getFeatures(messages,tokens,features_list,pos,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams,pos_trigrams,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,pos_trigrams_scores_objective,pos_trigrams_scores_subjective,pos_tags_scores_objective,pos_tags_scores_subjective,mpqaScores,negationList,clusters,pos_bigrams_scores_neutral=None,pos_trigrams_scores_neutral=None,pos_tags_scores_neutral=None):
    #initialize empty list with features for all message
    features = []

    #calculate features for every message
    for i in range(0,len(messages)):
        #list with features for one message
        f = calculateFeatures(messages[i],tokens[i],features_list,pos[i],slangDictionary,lexicons,mpqa_lexicons,pos_bigrams[i],pos_trigrams[i],pos_bigrams_scores_objective,pos_bigrams_scores_subjective,pos_trigrams_scores_objective,pos_trigrams_scores_subjective,pos_tags_scores_objective,pos_tags_scores_subjective,mpqaScores,negationList,clusters,pos_bigrams_scores_neutral,pos_trigrams_scores_neutral,pos_tags_scores_neutral)

        #add f to features
        features.append(f)

    #convert features list to numpy array
    features = np.array(features)
    #return test array , with no actual features
    #features_array = np.random.rand(len(messages),10)

    #return result
    return features
"""
calculate features for a message
"""
def calculateFeatures(message,tokens,features_list,pos,slangDictionary,lexicons,mpqa_lexicons,pos_bigrams,pos_trigrams,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,pos_trigrams_scores_objective,pos_trigrams_scores_subjective,pos_tags_scores_objective,pos_tags_scores_subjective,mpqaScores,negationList,clusters,pos_bigrams_scores_neutral,pos_trigrams_scores_neutral,pos_tags_scores_neutral):    

    f=[]
    #Morphological Features
  
    #existance of enlogated tokens in message e.g. "baaad"
    if 'has_elongated' in features_list:
        x = hasElongatedWords(message)
        f.append(x)

    #the number of elongated tokens in the message
    if 'num_elongated' in features_list:
        x = numberOfElongatedWords(message)
        f.append(x)

    #existance of date expressions in message
    if 'has_date' in features_list:
        x = hasDateExpressions(message)
        f.append(x)

    #existance of time expressions in message
    if 'has_time' in features_list:
        x = hasTimeExpressions(message)
        f.append(x)

    #the number of tokens of the message that are fully capitalized
    if 'num_all_caps' in features_list:
        x = countFullyCapitalizeTokens(tokens)
        f.append(x)

    #the number of tokens that are partially capitalized
    if 'num_part_caps' in features_list:
        x = countPartiallyCapitalizeTokens(tokens)
        f.append(x)

    #the number of tokens that start with an upper case letter
    if 'num_capitalized' in features_list: 
        x = countUpper(tokens)
        f.append(x)

    #the number of exclamation marks in the message
    ex = countExclamationMarks(message)
    if 'num_!' in features_list:
        f.append(ex)

    #the number of question marks

    qu = countQuestionMarks(message)
    if 'num_?' in features_list:
        f.append(qu)

    #the sum of exclamation and question marks
    if 'sum_!?' in features_list:
        x = ex + qu
        f.append(x)

    #the number of tokens containing only exclamation marks
    if 'num_tokens_only?' in features_list:
        x = onlyQuestionMarks(tokens)
        f.append(x)

    #the number of tokens containing only exclamation marks
    if 'num_tokens_only!' in features_list:
        x = onlyExclamationMarks(tokens)
        f.append(x)

    #the number of tokens containing only exclamation marks
    if 'num_tokens_only!?' in features_list:
        x = onlyQuestionOrExclamationMarks(tokens)
        f.append(x)

    #the number of tokens containing only ellipsis (...)
    if 'num_ellipsis' in features_list:
        x = countEllipsis(tokens)
        f.append(x)

    #the existence of a subjective emoticon at the message's end
    if 'ends_with_emoticon' in features_list:
        x = hasEmoticonAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
        f.append(x)

    #the existence of an ellipsis and a link (URL) at the message's end
    if 'ends_with_ellipsis_or_link' in features_list:
        x = hasUrlOrEllipsisAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
        f.append(x)

    #the existence of an exclamation mark at the message's end
    if 'ends_with_!' in features_list:
        x = hasExclamationMarkAtEnd(tokens[len(tokens)-1])
        f.append(x)

    #the existence of a question mark at the message's end
    if 'ends_with_?' in features_list:
        x = hasQuestionMarkAtEnd(tokens[len(tokens)-1])
        f.append(x)

    #the existence of a question or an exclamation mark at the message's end
    if 'ends_with_!?' in features_list:
        x = hasQuestionOrExclamationMarkAtEnd(tokens[len(tokens)-1])
        f.append(x)

    #the existence of slang
    if 'has_slang' in features_list:
        x = hasSlang(tokens,slangDictionary)
        f.append(x)

    #Pos Based Features

    #the number of adjectives in the message
    x1 = numberOfAdjectives(pos)
    #f.append(x)

    #the number of adverbs
    x2 = numberOfAdverbs(pos)
    #f.append(x)

    #the number of interjections
    x3 = numberOfIntejections(pos)
    #f.append(x)

    #the number of verbs
    x4 = numberOfVerbs(pos)
    #f.append(x)

    #the number of nouns
    x5 = numberOfNouns(pos)
    #f.append(x)

    #the number of proper nouns
    x6 = numberOfProperNouns(pos,tokens)
    #f.append(x)

    #the number of urls
    x7 = numberOfUrls(pos,tokens)
    #f.append(x)

    #the number of subjective emoticons
    x8 = numberOfSubjectiveEmoticons(pos,tokens)
    #f.append(x)

    #find the sum of "special" tokens
    s = x1+x2+x3+x4+x5+x6+x7+x8

    #divide scores with s and normialize to [-1,1]
    if 'num_adj' in features_list:
        f.append(2*(x1/float(s))-1)
    if 'num_adv' in features_list:
        f.append(2*(x2/float(s))-1)
    if 'num_intrj' in features_list:
        f.append(2*(x3/float(s))-1)
    if 'num_verb' in features_list:
        f.append(2*(x4/float(s))-1)
    if 'num_noun' in features_list:
        f.append(2*(x5/float(s))-1)
    if 'num_prop_noun' in features_list:
        f.append(2*(x6/float(s))-1)
    if 'num_urls' in features_list:
        f.append(2*(x7/float(s))-1)
    if 'num_emoticons' in features_list:
        f.append(2*(x8/float(s))-1)
        
##    f.append(x1/float(s))
##    f.append(x2/float(s))
##    f.append(x3/float(s))
##    f.append(x4/float(s))
##    f.append(x5/float(s))
##    f.append(x6/float(s))
##    f.append(x7/float(s))
##    f.append(x8/float(s))

    #Pos Tags Features
        
    #the average,maximun,minium f1 score for the messages pos bigrams for negative messages
    average, maximum, minimum = F1PosTagsScore(pos,pos_tags_scores_objective)
    if 'avg_f1_pos_unigrams_neg' in features_list:
        f.append(average)
    if 'max_f1_pos_unigrams_neg' in features_list:
        f.append(maximum)
    if 'min_f1_pos_unigrams_neg' in features_list:
        f.append(minimum)

    #the average,maximun,minium f1 score for the messages pos bigrams for positive messages
    average, maximum, minimum = F1PosTagsScore(pos,pos_tags_scores_subjective)
    if 'avg_f1_pos_unigrams_pos' in features_list:
        f.append(average)
    if 'max_f1_pos_unigrams_pos' in features_list:
        f.append(maximum)
    if 'min_f1_pos_unigrams_pos' in features_list:
        f.append(minimum)


    if pos_tags_scores_neutral is not None:
        #the average,maximun,minium f1 score for the messages pos bigrams for neutral messages
        average, maximum, minimum = F1PosTagsScore(pos,pos_tags_scores_neutral)
        if 'avg_f1_pos_unigrams_neutral' in features_list:
            f.append(average)
        if 'max_f1_pos_unigrams_neutral' in features_list:
            f.append(maximum)
        if 'min_f1_pos_unigrams_neutral' in features_list:
            f.append(minimum)



    #Pos Bigrams Features
    
    #the average,maximun,minium f1 score for the messages pos bigrams for objective messages
    
    
    average, maximum, minimum = F1PosBigramsScore(pos_bigrams,pos_bigrams_scores_objective)
    if 'avg_f1_pos_bigrams_neg' in features_list:
        f.append(average)
    if 'max_f1_pos_bigrams_neg' in features_list:
        f.append(maximum)
    if 'min_f1_pos_bigrams_neg' in features_list:
        f.append(minimum)


    #the average,maximun,minium f1 score for the messages pos bigrams for objective messages
    average, maximum, minimum = F1PosBigramsScore(pos_bigrams,pos_bigrams_scores_subjective)
    if 'avg_f1_pos_bigrams_pos' in features_list:
        f.append(average)
    if 'max_f1_pos_bigrams_pos' in features_list:
        f.append(maximum)
    if 'min_f1_pos_bigrams_pos' in features_list:
        f.append(minimum)
    
    if pos_bigrams_scores_neutral is not None:
        average, maximum, minimum = F1PosBigramsScore(pos_bigrams,pos_bigrams_scores_neutral)
        if 'avg_f1_pos_bigrams_neutral' in features_list:
            f.append(average)
        if 'max_f1_pos_bigrams_neutral' in features_list:
            f.append(maximum)
        if 'min_f1_pos_bigrams_neutral' in features_list:
            f.append(minimum)

    #Pos Trigrams Features

    #the average,maximun,minium f1 score for the messages pos bigrams for negative messages
    average, maximum, minimum = F1PosTrigramsScore(pos_trigrams,pos_trigrams_scores_objective)
    if 'avg_f1_pos_trigrams_neg' in features_list:
        f.append(average)
    if 'max_f1_pos_trigrams_neg' in features_list:
        f.append(maximum)
    if 'min_f1_pos_trigrams_neg' in features_list:
        f.append(minimum)

    #the average,maximun,minium f1 score for the messages pos bigrams for positive messages
    average, maximum, minimum = F1PosTrigramsScore(pos_trigrams,pos_trigrams_scores_subjective)
    if 'avg_f1_pos_trigrams_pos' in features_list:
        f.append(average)
    if 'max_f1_pos_trigrams_pos' in features_list:
        f.append(maximum)
    if 'min_f1_pos_trigrams_pos' in features_list:
        f.append(minimum)

    if pos_trigrams_scores_neutral is not None:
        average, maximum, minimum = F1PosBigramsScore(pos_trigrams,pos_trigrams_scores_neutral)
        if 'avg_f1_pos_trigrams_neutral' in features_list:
            f.append(average)
        if 'max_f1_pos_trigrams_neutral' in features_list:
            f.append(maximum)
        if 'min_f1_pos_trigrams_neutral' in features_list:
            f.append(minimum)
        
    # Lexicon Based Features
    lexicon_names=['minqinghu', 'afinn', 'nrc2', 'nrc5', 'nrc6', 'swn', 'swn_avg']
    #iterate for every lexicon
    for index, lexicon in enumerate(lexicons):
        #score of lexicon (total score of all words)
        if 'lex_score_'+lexicon_names[index] in features_list:
            x = sumOfScores(lexicon,message,tokens,pos)
            f.append(x)

        #average of scores
        if 'avg_lex_score_'+lexicon_names[index] in features_list:
            f.append(x/float(len(tokens)))

        #max score of words
        if 'max_lex_score_'+lexicon_names[index] in features_list:
            x = maxOfScores(lexicon,tokens,pos,False)
            f.append(x)

        #min score of words
        if 'min_lex_score_'+lexicon_names[index] in features_list:
            x = minOfScores(lexicon,tokens,pos,False)
            f.append(x)

        #the count of words of the message that appear in the lexicon
        if 'num_tokens_in_lex_'+lexicon_names[index] in features_list:
            x = numberOfAppearances(lexicon,tokens)
            f.append(x)

        #the score of the last word of the message
        if 'last_word_lex_score_'+lexicon_names[index] in features_list:
            x = scoreOfLastWord(lexicon,tokens[len(tokens)-1],pos[len(pos)-1])
            f.append(x)

        #the score of the last word of the message that appears in the lexicon
        if 'last_word_in_lex_score_'+lexicon_names[index] in features_list:
            x = scoreOfLastWordAppearedInLexicon(lexicon,tokens,pos)
            f.append(x)
    
    mpqa_lex_names=['S_pos','S_neg', 'S_pos_neg', 'S_neu', 
                    'W_pos', 'W_neg', 'W_pos_neg', 'W_neu', 
                    'semeval_neutral', 'semeval_positive', 'semeval_negative']
    #iterate for every mpqa lexicon (no score features because the mpqa lexicons have no subjectivity scores assinged to words)
    for index, lexicon in enumerate(mpqa_lexicons):
        #the count of words of the message that appear in the lexicon
        if 'num_words_in_lex_mpqa_' + mpqa_lex_names[index] in features_list:
            x = numberOfAppearances(lexicon,tokens)
            f.append(x)

    #lexicon presicion and F1 scores
    #lexicon_precision_objective, lexicon_f1_objective, lexicon_precision_subjective, lexicon_f1_subjective

    #iterate for every mpqa lexicon
    if pos_bigrams_scores_neutral is not None:
        step=6
    else:
        step=4
        
    for i in range(0,len(mpqaScores),step):
        name_idx = i % step
        #precision-objective
        average, minimum, maximum = LexiconScores(mpqaScores[i],tokens)
        
        if 'score_prec_obj_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(average)
        if 'score_prec_obj_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(minimum)
        if 'score_prec_obj_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(maximum)

        #precision-subjective
        average, minimum, maximum = LexiconScores(mpqaScores[i+1],tokens)
        if 'score_prec_subj_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(average)
        if 'score_prec_subj_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(minimum)
        if 'score_prec_subj_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(maximum)

        #F1-objective
        average, minimum, maximum = LexiconScores(mpqaScores[i+2],tokens)
        if 'score_f1_obj_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(average)
        if 'score_f1_obj_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(minimum)
        if 'score_f1_obj_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(maximum)

        #F1-subjective
        average, minimum, maximum = LexiconScores(mpqaScores[i+3],tokens)
        if 'score_f1_subj_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(average)
        if 'score_f1_subj_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(minimum)
        if 'score_f1_subj_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
            f.append(maximum)

        if pos_bigrams_scores_neutral is not None:
            #precision-neutral
            average, minimum, maximum = LexiconScores(mpqaScores[i+4],tokens)
            if 'score_prec_neutral_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(average)
            if 'score_prec_neutral_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(minimum)
            if 'score_prec_neutral_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(maximum)

            #F1-neutral
            average, minimum, maximum = LexiconScores(mpqaScores[i+5],tokens)
            if 'score_f1_neutral_avg_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(average)
            if 'score_f1_neutral_min_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(minimum)
            if 'score_f1_neutral_max_mpqa_' + mpqa_lex_names[name_idx] in features_list:
                f.append(maximum)
        

    #Other Features

    #check if message has negation
    if 'has_negation' in features_list:
        x = hasNegation(tokens,negationList)
        f.append(x)

    #check if message has negation preceding words from lexicon
    if 'has_negation_before_lex_S_pos_neg' in features_list:
        x = hasNegationPrecedingLexicon(mpqa_lexicons[2],tokens,negationList)
        f.append(x)
    
    if 'has_negation_before_lex_W_pos_neg' in features_list:
        x = hasNegationPrecedingLexicon(mpqa_lexicons[6],tokens,negationList)
        f.append(x)
    
    #Word Clusters
    if 'word_clusters' in features_list:
        tags = checkClusters(tokens,clusters)
        f+=tags

    return f

"""
reads a file of features to use in training/classification
inputs:
f - name of the file to load

outputs:
list of the names of features to use for training/classification
"""    
def load_feature_list(f):
    features_list = []
    with open(f, 'r') as f_file:
        for line in f_file:
            features_list.append(line.strip())
    return features_list