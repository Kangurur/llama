from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

model = "dkleczek/bert-base-polish-cased-v1"  # Polish BERT
tokenizer = AutoTokenizer.from_pretrained(model)

def encode_sentences(sentences):
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors="pt",padding=True, truncation=True,max_length=512)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Średnia po tokenach
# Dane treningowe: osadzenia zdań
#data="data/uj.csv"
with open("data/uj.csv", "r") as file:
    data = file.readlines()
    #data=np.array(data)

embeddings = encode_sentences(data).numpy()

# Standaryzacja danych
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Trening One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
svm.fit(scaled_embeddings)

# Ewaluacja na nowych danych
test_sentences = ["asdf", "Uniwersytet Jagielloński najlepszy.", "Papież lubi kremówki"]
test_embeddings = encode_sentences(test_sentences).numpy()
scaled_test_embeddings = scaler.transform(test_embeddings)

predictions = svm.predict(scaled_test_embeddings)
print(predictions)  # -1 oznacza anomalię, 1 oznacza normalne dane
