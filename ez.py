from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
import torch
#from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

tokenizer = AutoTokenizer.from_pretrained("olipol/smaug_part1")
model = AutoModelForSequenceClassification.from_pretrained("olipol/smaug_part1")
#tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
#base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
#model = PeftModel.from_pretrained(base_model, "olipol/smaug_test")
model.to('cpu') #daj na 'cpu' jak nie działa XD

n=int(input())
for i in range(n):
    text=input() #nie ogarnia jeszcze skrótu UJ
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu") 
    outputs = model(inputs)
    probs = outputs.logits.softmax(dim=-1)
    print("UJ" if probs[0][1] > probs[0][0] else "nie UJ")
    print(probs)
