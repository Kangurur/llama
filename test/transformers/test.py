from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

sequence = input()
xd=tokenizer(sequence)
print(xd)
print(tokenizer.decode(xd['input_ids']))