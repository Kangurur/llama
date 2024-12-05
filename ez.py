from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = PeftModel.from_pretrained(base_model, "olipol/smaug_test")
model.to('cuda') #daj na 'cpu' jak nie działa XD

n=int(input())
for i in range(n):
    text=input() #nie ogarnia jeszcze skrótu UJ
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda") 

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text , predictions.tolist()[0])
#tak w 80% powinien działać