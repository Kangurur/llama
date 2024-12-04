from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

data = load_dataset("data/uj")
#data=DataLoader(data)
model_id = "C:/Users/Oliwier/smaug/Llama-3.2-1B"
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
#classifier = pipeline("text-generation",model=model_id,device_map="auto")
def tokenizer_fn(x):
    return tokenizer(
        #x['sentence1'],
        #x['sentence2'],
        x['text'],
        padding="max_length",
        truncation=True,
        #max_length=128,
    )
    
tokenized_data = data.map(tokenizer_fn)
print(tokenized_data['train'][0])
#print(data)

