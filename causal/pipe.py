from transformers import pipeline
prompt="prezesem ksi jest"
generator = pipeline("text-generation", model="username/my_awesome_eli5_clm-model")

generator(prompt)