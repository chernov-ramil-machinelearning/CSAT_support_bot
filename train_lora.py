import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weight_model"
DATA_PATH = BASE_DIR / "dataset_ml_support.jsonl"
OUTPUT_DIR = BASE_DIR / "lora_output"

def main():
    # Проверяем наличие токенизатора и весов
    has_local_weights = MODEL_PATH.exists() and any(MODEL_PATH.iterdir())
    env_model = os.getenv("MODEL_PATH")

    if env_model:
        model_source = env_model
        local_only = False
    elif has_local_weights:
        model_source = str(MODEL_PATH)
        local_only = True
    else:
        print("=" * 72)
        print("ОШИБКА: Локальные веса модели не найдены в папке 'weight_model/'.")
        print("\nКаждый пользователь может использовать собственную модель:")
        print("  1. Поместите файлы вашей модели и токенизатора в папку 'weight_model/'.")
        print("  2. Либо скачайте рекомендованную базовую модель Qwen2.5-0.5B-Instruct:")
        print("     huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir weight_model")
        print("  3. Либо укажите имя модели в переменной окружения: MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct")
        print("=" * 72)
        return

    print(f"Загрузка токенизатора и модели из {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем модель в float16 на GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_only
    )

    # Настройка LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Загрузка датасета
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    system_prompt = (
        "Ты русскоязычный ассистент по основам машинного обучения (ML). "
        "Твоя задача — кратко, понятно и строго по делу объяснять алгоритмы, метрики, валидацию и подготовку данных в ML. "
        "Если вопрос не относится к машинному обучению (политика, быт, оффтоп) или содержит оскорбления и нецензурную брань, "
        "вежливо откажись отвечать в 1-2 предложениях и предложи задать вопрос по машинному обучению."
    )

    def preprocess(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = tokenizer(prompt, max_length=512, truncation=True)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    )

    print("\nНачинаем дообучение LoRA на GPU...")
    trainer.train()

    print(f"\nСохранение LoRA адаптера в {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Готово!")


if __name__ == "__main__":
    main()
