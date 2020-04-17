import main

import json
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello_world():
	try:
		filename = request.form['filename']
	except KeyError:
		filename = "vivo.csv"

	features, result = main.get_features_and_classification(filename)
	
	result_json = []
	for i, row in result.iterrows():
		result_json.append({
			"category": row['category'],
			"sentence": row['sentence'],
			"sentiment": row['sentiment']
		})

	final = {
		"features": features,
		"classification": result_json
	}

	return final

print("Server started")