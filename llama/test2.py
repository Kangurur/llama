import torch
from transformers import pipeline

model_id = "Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    #model_kwargs={"torch_dtype":"torch.bfloat16"},
    device_map="auto"
)
output = pipe(
    'ile żyje kangur?\n',
    do_sample=True,
    top_k=10,
    #num_return_sequences=1,
    #eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
    #max_new_tokens=256,
)
#message = [{"role":"user","content":"the capital of poland is"},]
#output=pipe(message, max_length=50, do_sample=False,max_new_tokens=256,)


print(output[0]['generated_text'])