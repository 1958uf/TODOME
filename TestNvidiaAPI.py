from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="qwen/qwen2.5-coder-32b-instruct",
  messages=[{"role":"user","content":""}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)

