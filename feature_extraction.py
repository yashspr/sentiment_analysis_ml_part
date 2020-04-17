import numpy as np
import pandas as pd
import enchant
from spellchecker import SpellChecker
from spacy.matcher import Matcher
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

d = enchant.Dict("en_US")
spell = SpellChecker(distance=1)

def feature_extraction(df, ft_model, nlp):
	all_nouns = []

	for review in df['spacyObj']:
		for token in review:
			if token.pos_ == "NOUN":
				all_nouns.append(token.text)

	all_nouns = pd.Series(all_nouns)
	unique_nouns = all_nouns.value_counts()

	for key in unique_nouns.keys():
		if not d.check(key):
			del unique_nouns[key]

	noun_phrases = []

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

	unique_nouns_list = list(unique_nouns.index)

	for noun in unique_nouns_list:
		new_spelling = spell.correction(noun)
		if new_spelling != noun:
			# Then we have detected a spelling error
			count = unique_nouns[noun]
			try:
				unique_nouns[new_spelling] += count
			except:
				unique_nouns[new_spelling] = count
			del unique_nouns[noun]

	for noun in unique_nouns.index:
		if len(noun) < 3:
			del unique_nouns[noun]

	top2 = int(len(unique_nouns) * 0.02)
	top_features = unique_nouns[0:top2]

	top_features_list = list(top_features.keys())
	unique_noun_phrases_list = list(unique_noun_phrases.keys())

	features_bucket = OrderedDict()

	for feature1 in top_features.keys():
		for feature2 in top_features.keys():
			feature_phrase = feature1 + ' ' + feature2
			if feature_phrase in unique_noun_phrases_list and feature1 in top_features_list and feature2 in top_features_list:

				lesser_occurring_noun = ""
				often_occurring_noun = ""
				if unique_nouns[feature1] < unique_nouns[feature2]:
					lesser_occurring_noun = feature1
					often_occurring_noun = feature2
				else:
					lesser_occurring_noun = feature2
					often_occurring_noun = feature1
				
				if unique_noun_phrases[feature_phrase]/unique_nouns[lesser_occurring_noun] > 0.3:
					try:
						if often_occurring_noun not in features_bucket:
							features_bucket[often_occurring_noun] = []
						features_bucket[often_occurring_noun].append(lesser_occurring_noun)
						top_features_list.remove(lesser_occurring_noun)
						# print(lesser_occurring_noun)
					except BaseException as error:
						print(error)
						continue

	top_features_copy = list(top_features_list)

	for feature1 in top_features_copy:
		if feature1 not in features_bucket:
			features_bucket[feature1] = []
		
		for feature2 in top_features_copy:
			similarity =  cosine_similarity(ft_model.get_word_vector(feature1).reshape(1, -1), 
										ft_model.get_word_vector(feature2).reshape(1, -1))
			if similarity[0][0] <= 0.99 and similarity[0][0] > 0.64:
				features_bucket[feature1].append(feature2)
				if feature2 in features_bucket:
					features_bucket[feature1] = features_bucket[feature1] + features_bucket[feature2]
					del features_bucket[feature2]
				top_features_copy.remove(feature2)

	return features_bucket