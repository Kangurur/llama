from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
import torch

torch.cuda.empty_cache()
model_id="distilbert/distilgpt2"
tokenizer=AutoTokenizer.from_pretrained(model_id,padding_side="right")
model=AutoModelForCausalLM.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"])

#dataset_train = dataset_train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#dataset_val = dataset_val.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
dataset=Dataset.from_json("bedziedzialac/causal/ujeng.jsonl")
dataset=dataset.map(tokenize_function, batched=True, remove_columns=["text","link"])
#print(dataset_train[0])

#print(dataset)

block_size = 64
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset=dataset.map(group_texts, batched=True)

training_args = TrainingArguments(
    output_dir="bedziedzialac/causal/test",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    #save_steps=10_000,
    #save_total_limit=2,
    optim="sgd",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
tokenizer.pad_token = tokenizer.eos_token
#dataset.train_test_split(test_size=0.1)

dataset=dataset.train_test_split(test_size=0.1)
#dataset=dataset[:-1]

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=dataset,
    #eval_dataset=dataset_test,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False),
    tokenizer=tokenizer,
)

trainer.train()
model_inputs = tokenizer(["Collegium Maius is the oldest"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs,max_new_tokens=100)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
#print (dataset[0])
#do testu