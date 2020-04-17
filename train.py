import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from preprocess import preprocess

def get_sigle_aspect_reviews(*dfs):
	#count reviews that talk about only one aspect
	total_count = 0
	reviews = []
	ratings = []

	all_features = ['android', 'battery', 'camera', 'charger', 'charging', 'delivery', 'device', 'display', 'features', 'fingerprint', 'gaming', 'issue', 'mode', 'money', 'performance', 'phone', 'price', 'problem', 'product', 'screen']

	for df in dfs: 
		for i, review in df['spacyObj'].items():
			flag = True
			found = ""

			for token in review:
				if token.text in all_features:
					if found == "":
						found = token.text
					elif found != token.text:
						flag = False
						break

			# if found == "":
			#     flag = False

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

def get_model():

	if os.path.isfile('model.joblib') and os.path.isfile('tfidf.joblib'):
		print("Trained models found. Using them.")
		model = load('model.joblib')
		tfidf = load('tfidf.joblib')

	else:
		print("Trained models not found. Training now!")
		nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])

		train_data = pd.read_csv('data.csv', header=None, names=['reviewText', 'rating'])
		train_data = preprocess(train_data)
		train_data['spacyObj'] = train_data['reviewText'].apply(lambda x: nlp(x))

		single_aspect_reviews = get_sigle_aspect_reviews(train_data)

		X_train = single_aspect_reviews['reviewText']
		y_train = single_aspect_reviews['rating'].apply(lambda x: giveRating(x))

		tfidf = TfidfVectorizer()

		X_train_tfidf = tfidf.fit_transform(X_train)

		model = RandomForestClassifier(n_estimators=100)
		model.fit(X_train_tfidf, y_train)

		dump(model, 'model.joblib')
		dump(tfidf, 'tfidf.joblib')

	return model, tfidf