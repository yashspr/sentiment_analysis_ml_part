import numpy as np
import pandas as pd
import re
from spellchecker import SpellChecker
from spacy.matcher import Matcher
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def feature_extraction(df, ft_model, nlp):    
    # Extracting all the single nouns in the corpus
    all_nouns = []

    for review in df['spacyObj']:
        for token in review:
            if token.pos_ == "NOUN":
                all_nouns.append(token.text)
        
    all_nouns = pd.Series(all_nouns)
    # Finding unique nouns along with their counts sorted in descending order
    unique_nouns = all_nouns.value_counts()

    noun_phrases = []
    
    # Pattern to match i.e. two nouns occuring together
    patterns = [
        [{'TAG': 'NN'}, {'TAG': 'NN'}]
    ]

    matcher = Matcher(nlp.vocab)
    matcher.add('NounPhrasees', patterns)

    for review in df['spacyObj']:
        matches = matcher(review)

        for match_id, start, end in matches:
            noun_phrases.append(review[start:end].text)
            
    noun_phrases = pd.Series(noun_phrases)
    unique_noun_phrases = noun_phrases.value_counts()
            
    # Remove nouns with single or double character
    for noun in unique_nouns.index:
        # if noun length is less than 3 or if nouns contain any numbers, it is considered invalid
        if len(noun) < 3 or re.match(r".*[0-9].*", noun) is not None:
            del unique_nouns[noun]
            
    # Extracting Top Features
    
    top2 = len(unique_nouns)*0.05 # considering top 5% of features
    top2 = int(top2)
    
    top_features = unique_nouns[0:top2]
    
    # this will contain all the final features
    features_bucket = OrderedDict()
    
    top_features_list = list(top_features.keys())
    top_features_set = set(top_features.keys())
    unique_noun_phrases_set = set(unique_noun_phrases.keys())
    
    # Applying assocation rule mining to group nouns occuring together
    for feature1 in top_features_list:
        for feature2 in top_features_list:
            feature_phrase = feature1 + ' ' + feature2
            
            if feature1 in top_features_set and feature2 in top_features_set and feature_phrase in unique_noun_phrases_set:
                # If the condition is true, we have identified a noun phrase which is a combination of two nouns
                # in the top_features. So one of the nouns cn be eliminated from top features.

                # Ex. if "battery life" is found, then "life" can be eliminated from top features as it is not a feature 
                # by itself. It is just part of the feature "battery life"

                # Now we need to find out if frequency of the lesser occuring noun (in our ex., the word "life") matches
                # with the frequency of the noun phrase (in our ex., "battery life") by a certain confidence. 
                # If it does so, then we can be sure that the lesser occuring noun occurs only in that particular noun_phrase
                # i.e in our ex "life" occurs primaryly in the phrase "battery life"

                lesser_occurring_noun = ""
                often_occurring_noun = ""
                if unique_nouns[feature1] < unique_nouns[feature2]:
                    lesser_occurring_noun = feature1
                    often_occurring_noun = feature2
                else:
                    lesser_occurring_noun = feature2
                    often_occurring_noun = feature1

                # assuming confidence interval of 40%
                # i.e. accordnig to 'battery life' example, out of total times that 'life' is seen, 'battery' is seen next to it 40% of the time. 

                if unique_noun_phrases[feature_phrase]/unique_nouns[lesser_occurring_noun] > 0.4:
                    try:
                        if often_occurring_noun not in features_bucket:
                            features_bucket[often_occurring_noun] = []
                        features_bucket[often_occurring_noun].append(lesser_occurring_noun)
                        top_features_set.remove(lesser_occurring_noun)
                        # print(lesser_occurring_noun)
                    except BaseException as error:
                        print(error)
                        continue
    
    main_features = list(features_bucket.keys())
    top_features_to_add = set(top_features_list[:20])
    
    # here we are manually adding adding 20 top nouns as features which were previously not
    # added by the assocation rule mining step above.
    # But before adding, we are checking if any similar nouns exist among the 20 nouns.
    # Ex. If 'display' and 'screen' occur in the top 20, we must add only the most commonly occuring
    # one among the two and remove the other.

    # Here we are only eliminating the nouns that are similar to existing ones in features_bucket.
    for feature1 in top_features_list[:20]:
        for feature2 in main_features:
            if feature1 not in features_bucket and feature1 in top_features_set:
                similarity =  cosine_similarity(ft_model.get_word_vector(feature1).reshape(1, -1), 
                                                   ft_model.get_word_vector(feature2).reshape(1, -1))
                if similarity[0][0] > 0.64:
                    top_features_to_add.discard(feature1)

            else:
                top_features_to_add.discard(feature1)

    top_features_to_add_list = list(top_features_to_add)

    # Here we are eliminating nouns that are similar to one another in the top_features_to_add
    for feature1 in top_features_to_add_list:
        for feature2 in top_features_to_add_list:
            if feature1 in top_features_to_add and feature2 in top_features_to_add:
                similarity =  cosine_similarity(ft_model.get_word_vector(feature1).reshape(1, -1), 
                                                   ft_model.get_word_vector(feature2).reshape(1, -1))
                if similarity[0][0] < 0.99 and similarity[0][0] > 0.64:
                    feature_to_remove = min((unique_nouns[feature1], feature1), (unique_nouns[feature2], feature2))[1]
                    top_features_to_add.remove(feature_to_remove)

    for feature in top_features_to_add:
        features_bucket[feature] = []
        
    for main_noun in features_bucket.keys():
        top_features_set.remove(main_noun)
        
    # Here we are going through the top 5% of the nouns that we originally considering and checking
    # if any of them are similar to the ones already present in features_bucket.    
    top_features_copy = list(top_features_set)
    main_features = features_bucket.keys()

    for feature2 in top_features_copy:
        best_similarity = 0
        most_matching_main_feature = ""

        for feature1 in main_features:
            if feature2 in top_features_set:
                similarity =  cosine_similarity(ft_model.get_word_vector(feature1).reshape(1, -1), 
                                               ft_model.get_word_vector(feature2).reshape(1, -1))
                if similarity[0][0] <= 0.99 and similarity[0][0] > 0.62:
                    if similarity[0][0] > best_similarity:
                        best_similarity = similarity[0][0]
                        most_matching_main_feature = feature1

        if best_similarity != 0 and most_matching_main_feature != "":       
            features_bucket[most_matching_main_feature].append(feature2)
            top_features_set.remove(feature2)

    # We finally sort the features in descending order based on how often they occur.        
    final_features = list(features_bucket.items())
    
    final_features_with_counts = []
    for feature in final_features:
        count = unique_nouns[feature[0]]
        final_features_with_counts.append((feature, count))

    final_features_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    final_features = OrderedDict()
    for feature, count in final_features_with_counts:
        final_features[feature[0]] = feature[1]
    
    return final_features