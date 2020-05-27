import re
import pandas as pd
import numpy as np

import constants

appos = constants.appos

def construct_spacy_obj(df, nlp):
	# constructing spacy object for each review
	with nlp.disable_pipes(['parser', 'ner']):
		docs = list(nlp.pipe(df['reviewText']))
		df['spacyObj'] = pd.Series(docs, index=df['reviewText'].index)
	
	return df

def reduce_lengthening(text):
	pattern = re.compile(r"([a-zA-Z])\1{2,}")
	return pattern.sub(r"\1\1", text)

def preprocess_text(txt, nlp):
	txt = txt.lower()
	txt = reduce_lengthening(txt)
	with nlp.disable_pipes('tagger', 'parser', 'ner', 'sentencizer'):
		doc = nlp(txt)
	tokens = [token.text for token in doc]
	
	if len(tokens) <3:
		return np.NaN
	
	for i, token in enumerate(tokens):
		if token in appos:
			tokens[i] = appos[token]
			
	txt = ' '.join(tokens)
	txt = re.sub(r"[^a-zA-Z0-9.,:;\-'?!/\n]", " ", txt)
	txt = re.sub(r"\n", ".", txt)
	txt = re.sub(r" ([.,:?;])", r"\1", txt)
	txt = re.sub(r"([. ])\1{1,}", r"\1", txt)
	
	return txt

def preprocess(df, nlp):
	df.dropna(inplace=True)
	df['reviewText'] = df['reviewText'].apply(lambda x: preprocess_text(x, nlp))
	df.dropna(inplace=True)

	return df