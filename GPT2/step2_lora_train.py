import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# ===== 1. 加载模型 =====
model_name = r"D:\AI-explorer\models\tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# ===== 2. 配置 LoRA =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== 3. 读取数据 =====
with open("data.json", "r") as f:
    raw_data = json.load(f)

def preprocess(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    full_text = prompt + " " + example["output"]

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    # 🔑 关键：labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


dataset = [preprocess(x) for x in raw_data]

# ===== 4. 训练参数 =====
training_args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ===== 5. 开始训练 =====
trainer.train()

# ===== 6. 保存 LoRA 权重 =====
model.save_pretrained("lora_adapter")
