from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer
import torch
from datasets import load_dataset

#data=load_dataset("PenguinPush/animals-info")
#data=load_dataset("glue",'mrpc')
data = load_dataset("yelp_review_full")
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

#text=["i love kangaroos"]
#print(classifier(text))
#print(type(AutoModel.from_pretrained(model_id)))
#print(AutoConfig.from_pretrained(model_id))
#print(data['train'])
print(data['train'][0])
tokenized_data = data.map(tokenizer_fn)
print(tokenized_data.column_names)

