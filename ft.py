import os
import fasttext

def get_model():
	if os.path.isfile('models/fasttext_model_cbow.bin'):
		print("FastText model already exists")
		model = fasttext.load_model("models/fasttext_model_cbow.bin")
	else:
		print("FastText model doesn't exist. Training and saving it.")
		model = fasttext.train_unsupervised('csv_files/fasttext_data.txt', model='cbow')
		model.save_model("models/fasttext_model_cbow.bin")
	
	return model
