import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

import constants
from feature_extraction import feature_extraction

# contains mapping such as "don't" => "do not"
appos = constants.appos
stopwords = constants.stopwords

scoring = ['accuracy', 'precision_macro', 'recall_macro']

# normalizing exaggerated words
def reduce_lengthening(text):
	pattern = re.compile(r"([a-zA-Z])\1{2,}")
	return pattern.sub(r"\1\1", text)

def preprocess(txt, nlp):
	txt = txt.lower() # converting text to lower case
	txt = reduce_lengthening(txt) # normalizing exaggerated words
	
	with nlp.disable_pipes('tagger', 'parser', 'ner'):
		doc = nlp(txt) # tokenizing the words
	
	tokens = [token.text for token in doc]
	
	# removing reviews with less than 3 tokens
	if len(tokens) <3:
		return np.NaN
	
	# normalizing words with apostrophe
	for i, token in enumerate(tokens):
		if token in appos:
			tokens[i] = appos[token]
			
	txt = ' '.join(tokens)
	txt = re.sub(r"[^a-zA-Z. \n]", " ", txt)
	txt = re.sub(r"([. \n])\1{1,}", r"\1", txt)
	txt = re.sub(r" ([.\n])", r"\1", txt)
	txt = re.sub(r" ?\n ?", ".", txt)
	txt = re.sub(r"([. \n])\1{1,}", r"\1", txt)
	
	return txt.strip()

def postprocess(x, nlp):
	# removing stop words
	with nlp.disable_pipes(['tagger', 'parser', 'ner', 'sentencizer']):
		doc = nlp(x)
	
	words = [token.text for token in doc if token.text not in stopwords]
	x = ' '.join(words)
	x = re.sub(r"[0-9\n.?:;,-]", " ", x)
	x = re.sub(r"[ ]{2,}", " ", x)
	
	return x

def construct_spacy_obj(df, nlp):
	with nlp.disable_pipes(['parser', 'ner', 'sentencizer']):
		# constructing spacy object for each review
		docs = list(nlp.pipe(df['reviewText']))
		df['spacyObj'] = pd.Series(docs, index=df['reviewText'].index)
	
	return df

def get_sigle_aspect_reviews(*dfs, features):
	#count reviews that talk about only one aspect
	total_count = 0
	reviews = []
	ratings = []

	# all_features = ['android', 'battery', 'camera', 'charger', 'charging', 'delivery', 'device', 'display', 'features', 'fingerprint', 'gaming', 'issue', 'mode', 'money', 'performance', 'phone', 'price', 'problem', 'product', 'screen']

	for df in dfs: 
		for i, review in df['spacyObj'].items():
			flag = True
			found = set()

			for token in review:
				if token.text in features:
					if len(found) <3:
						found.add(token.text)
					elif token.text not in found:
						flag = False
						break

			if flag:
				total_count += 1
				reviews.append(review.text)
				ratings.append(df['rating'][i])
	
	print(total_count)
	return pd.DataFrame({'reviewText': reviews, 'rating': ratings})

def giveRating(x):
	if x in [5,4]:
		return "Positive"
	elif x in [1,2,3]:
		return "Negative"

def get_model(nlp, ft_model):

	if os.path.isfile('models/model.joblib'):
		print("Trained model found. Using them.")
		model = load('models/model.joblib')
		# tfidf = load('models/tfidf.joblib')

	else:
		print("Trained models not found. Training now!")

		train_data = pd.read_csv('csv_files/training.csv', header=None, names=['reviewText', 'rating'])
		train_data.dropna(inplace=True)
		train_data['reviewText'] = train_data['reviewText'].apply(lambda x: preprocess(x, nlp))
		train_data.dropna(inplace=True)
		train_data = construct_spacy_obj(train_data, nlp)

		features = feature_extraction(train_data, ft_model, nlp)

		single_aspect_reviews = get_sigle_aspect_reviews(train_data, features=features)
		single_aspect_reviews['reviewText'] = single_aspect_reviews['reviewText'].apply(lambda x: postprocess(x, nlp))

		X_train = single_aspect_reviews['reviewText']
		y_train = single_aspect_reviews['rating'].apply(lambda x: giveRating(x))

		final_lr = Pipeline([
			('tfidf', TfidfVectorizer(lowercase=False, min_df=0.00006, ngram_range=(1,3))),
			('lr', LogisticRegression(solver='lbfgs', max_iter=175))
		])

		# final_rf = Pipeline([
		# 	('tfidf', TfidfVectorizer(lowercase=False, min_df=0.00006, ngram_range=(1,3))),
		# 	('rf', RandomForestClassifier(n_estimators=100))
		# ])

		scores_final_lr = cross_validate(final_lr, X_train, y_train, scoring=scoring, cv=5)

		for scoring_measure, scores_arr in scores_final_lr.items():
			print(scoring_measure, ":\t%f (+/- %f)" % (scores_arr.mean(), scores_arr.std()*2))

		# scores_final_rf = cross_validate(final_rf, X_train, y_train, scoring=scoring, cv=5)

		# for scoring_measure, scores_arr in scores_final_rf.items():
		# 	print(scoring_measure, ":\t%f (+/- %f)" % (scores_arr.mean(), scores_arr.std()*2))

		final_lr.fit(X_train, y_train)
		# final_rf.fit(X_train, y_train)

		dump(final_lr, 'models/model.joblib')
		# dump(final_rf, 'models/model_rf.joblib')
		# dump(tfidf, 'tfidf.joblib')

		model = final_lr

	return model