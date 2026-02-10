from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = r"D:\AI-explorer\models\tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# 加载 LoRA adapter
model = PeftModel.from_pretrained(model, "lora_adapter")

prompt = "The meaning of life is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=40,
    do_sample=False
)

print(tokenizer.decode(outputs[0]))
