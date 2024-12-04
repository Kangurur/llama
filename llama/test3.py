import torch
from transformers import pipeline

model_id = "Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    #model_kwargs={"torch_dtype":"torch.bfloat16"},
    device_map="auto",
    max_new_tokens=150
)

output=pipe("prezesem Koła Informatyków UJ jest ",num_return_sequences=1,pad_token_id=pipe.tokenizer.eos_token_id)

print(output)