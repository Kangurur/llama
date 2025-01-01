from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.2-1b-instruct",
  messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

