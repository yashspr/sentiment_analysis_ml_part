import pandas as pd
from feature_extraction import feature_extraction

def construct_rev_lookup(features):
	bucket_lookup = {}

	for bucket, similar_features in features.items():
		bucket_lookup[bucket] = bucket

		for feature in similar_features:
			if feature not in bucket_lookup:
				bucket_lookup[feature] = bucket
					
	return bucket_lookup

def classify(df, features, model):
	lookup = construct_rev_lookup(features)

	features = []
	sub_categories = []
	sentences = []
	sentiments = []
	
	no_cat_sents = []
	more_than_one_sents = []
	
	for review in df['spacyObj']:
		for sent in review.sents:
			
			# lets check if the sentence contains more than one token
			no_of_valid_tokens = 0
			for token in sent:
				if token.is_alpha:
					no_of_valid_tokens += 1
					
			if no_of_valid_tokens < 2:
				continue
				
			cat = None
			sub_cat = None
			flag = True
			
			for token in sent:
				if token.text in lookup:
					if cat is None:
						cat = lookup[token.text]
						sub_cat = token.text
						
					elif cat is not None and lookup[token.text] != cat:
						flag = False
						more_than_one_sents.append(sent.text)
						break
						
			if cat == None:
				flag = False
				no_cat_sents.append(sent.text)

			if flag:
				# Now we know the sentence contains only one feature and find sentiment of that
				pred = model.predict([sent.text])[0]
				features.append(cat)
				sub_categories.append(sub_cat)
				sentences.append(sent.text)
				sentiments.append(pred)
				
	results_df = pd.DataFrame({'category': features, 'sub_category': sub_categories, 'sentence': sentences, 'sentiment': sentiments})
	no_cat_df = pd.DataFrame({'sentence': no_cat_sents})
	more_than_one_df = pd.DataFrame({'sentences': more_than_one_sents})
	
	return results_df, more_than_one_df, no_cat_df