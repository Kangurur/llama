from transformers import pipeline

# Ładowanie wytrenowanego modelu
qa_pipeline = pipeline("text2text-generation", model="./testXD/trained_model", tokenizer="./testXD/trained_model")

# Testowe pytania
question = "How many positive two-digit integers have an odd number of positive factors?"
result = qa_pipeline(question, max_length=100, num_return_sequences=1, truncation=True)
print(question)
print(result[0]['generated_text'])

