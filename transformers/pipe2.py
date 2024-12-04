from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer
import torch
from datasets import load_dataset



#model_id = "C:/Users/Oliwier/smaug/Llama-3.2-1B"
model_id="C:/Users/Oliwier/smaug/smaug_test/checkpoint-200"

classifier = pipeline("sentiment-analysis",model=model_id)


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

#n=int(input())
for i in text_list:
    text=i
    print(classifier(text))