from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
import torch
from datasets import load_dataset

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig


#model_id = "C:/Users/Oliwier/smaug/Llama-3.2-1B"
#model_id="C:/Users/Oliwier/smaug/smaug_test/checkpoint-200"
#model = AutoModel.from_pretrained("olipol/smaug_test")
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
#classifier = pipeline("sentiment-analysis",model="olipol/smaug_test",)


#text=["lubie Uniwersytet Jagielloński"]
#text=input()
#print(classifier(text))
#print(type(AutoModel.from_pretrained(model_id)))
#print(AutoConfig.from_pretrained(model_id))
#print(data['train'])
text_list=["Lubię kangury",
           "Uniwersytet Jagielloński jest najlepszy",
           "Dlaczego jestem upośledzony?",
           "Błagam pomocy",
           "Gdzie jest uniwersytet jagielloński?",
           "Czy mogę dostać się na uniwersytet jagielloński?",
           "Czy uniwersytet jagielloński posiada koło studentów informatyki?",
           "Co sądzisz o papieżu?",
           "Jak nazywa się dziekan wydziału informatyki?",
           "Czemu zużywasz tyle ramu?"
           ]

n=int(input())

base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = PeftModel.from_pretrained(base_model, "olipol/smaug_test")

model.to('cuda')
for i in range(n):
    text=input()
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text , predictions.tolist()[0])