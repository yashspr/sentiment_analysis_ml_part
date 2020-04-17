import os
import fasttext

def get_model():
	if os.path.isfile('./fasttext_model_cbow.bin'):
		print("FastText model already exists")
		model = fasttext.load_model("./fasttext_model_cbow.bin")
	else:
		print("FastText model doesn't exist. Training and saving it.")
		model = fasttext.train_unsupervised('fasttext_data.txt', model='cbow')
		model.save_model("fasttext_model_cbow.bin")
	
	return model
