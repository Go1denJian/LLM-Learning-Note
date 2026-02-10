from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = r"D:\AI-explorer\models\tiny-gpt2"

# 1. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 输入 prompt
prompt = "The meaning of life is"

inputs = tokenizer(prompt, return_tensors="pt")

# 3. 生成文本
outputs = model.generate(
    **inputs,
    max_length=30,
    do_sample=True,
    temperature=0.8
)

# 4. 解码输出
result = tokenizer.decode(outputs[0])
print("!!!THIS IS RESULT: " + result)
