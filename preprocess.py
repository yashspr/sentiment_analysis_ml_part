import re

def reduce_lengthening(text):
	pattern = re.compile(r"(.)\1{2,}")
	return pattern.sub(r"\1\1", text)

def preprocess(df):
	df.dropna(inplace=True)
	df['reviewText'] = df['reviewText'].apply(lambda x: x.lower())
	df['reviewText'] = df['reviewText'].apply(lambda x: reduce_lengthening(x))

	return df