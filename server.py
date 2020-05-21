import main

import json
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def classify():
	try:
		filename = request.form['filename']
	except KeyError:
		filename = "vivo.csv"

	features, result = main.get_features_and_classification(filename)
	final_features = {}

	for feature in features.keys():
		final_features[feature] = {
			"related": features[feature],
		}
		
		try:
			final_features[feature]["positives"] = str(result[result["category"] == feature]["sentiment"].value_counts()['Positive'])
		except KeyError:
			final_features[feature]["positives"] = 0

		try:
			final_features[feature]["negatives"] = str(result[result["category"] == feature]["sentiment"].value_counts()['Negative'])
		except KeyError:
			final_features[feature]["negatives"] = 0
	
	result_json = []
	for i, row in result.iterrows():
		result_json.append({
			"category": row['category'],
			"sentence": row['sentence'],
			"sentiment": row['sentiment']
		})

	final = {
		"features": final_features,
		"classification": result_json,
		"productID": filename.split('.')[0]
	}

	return final

print("Server started")