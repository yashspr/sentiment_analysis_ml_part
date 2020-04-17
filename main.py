import spacy
from spacy.pipeline import Sentencizer
import pandas as pd

from preprocess import preprocess
import ft
from feature_extraction import feature_extraction
from classifiation import classify
import train

ft_model = ft.get_model()
model, tfidf = train.get_model()

nlp = spacy.load('en_core_web_sm', disable=['parser','ner','textcat'])
sentencizer = Sentencizer(punct_chars=[".", "!", "?", "..", "...", "\n", "\r", ":", ";"])
nlp.add_pipe(sentencizer)

def construct_spacy_obj(df):
	# constructing spacy object for each review
	df['spacyObj'] = df['reviewText'].apply(lambda x: nlp(x))
	
	return df

def get_features_and_classification(filename):
	df = pd.read_csv(filename, header=None, names=['reviewText', 'rating'])
	df = preprocess(df)
	df = construct_spacy_obj(df)

	features = feature_extraction(df, ft_model, nlp)
	result, _, __ = classify(df, features, model, tfidf)

	return features, result