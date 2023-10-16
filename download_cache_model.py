import os
import sys

model_name = sys.argv[1]

if not os.path.exists("./pretrained"):
		os.makedirs("./pretrained")

# download model
from transformers import AutoTokenizer, AutoModel

my_tokenizer = AutoTokenizer.from_pretrained(model_name)
my_model = AutoModel.from_pretrained(model_name)

my_model.save_pretrained("./pretrained/" + model_name)
my_tokenizer.save_pretrained("./pretrained/" + model_name)