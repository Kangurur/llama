from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import BitsAndBytesConfig
import torch

torch.cuda.empty_cache()
model_id="meta-llama/Llama-3.2-3B"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"])

#dataset_train = dataset_train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#dataset_val = dataset_val.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
dataset=Dataset.from_json("wiki3.jsonl")
dataset=dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#print(dataset_train[0])

block_size = 32
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
#print(dataset[0])

training_args = TrainingArguments(
    output_dir="test",
    eval_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=30,
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    optim="sgd",
)
tokenizer.pad_token = tokenizer.eos_token
#dataset.train_test_split(test_size=0.1)

dataset_test=[]
dataset_test.append(dataset[-1])
#dataset=dataset[:-1]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_test,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

trainer.train()
model_inputs = tokenizer(["Członkowie zarządu KSI to"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs,max_new_tokens=100)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
#print (dataset[0])
#do testu