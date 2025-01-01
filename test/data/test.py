from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
eli5 = load_dataset("eli5_category", split="train[:5000]")
eli5=eli5.flatten()
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])
tokenized_eli5 = eli5.map(

    remove_columns=['q_id', 'title', 'selftext', 'category', 'subreddit', 'answers.a_id', 'answers.score', 'answers.text_urls', 'title_urls', 'selftext_urls']
)

with open("test/data/rozmiar.jsonl", "w") as f:
    for i in tokenized_eli5:
        f.write(str(i) + "\n")