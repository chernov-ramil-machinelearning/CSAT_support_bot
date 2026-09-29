import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
LOCAL_WEIGHTS = BASE_DIR / "weight_model"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Проверяем наличие токенизатора в локальной папке
if LOCAL_WEIGHTS.exists() and any(LOCAL_WEIGHTS.iterdir()):
    model_source = str(LOCAL_WEIGHTS)
elif os.getenv("MODEL_PATH"):
    model_source = os.getenv("MODEL_PATH")
else:
    print("=" * 72)
    print("ОШИБКА: Локальный токенизатор не найден в папке 'weight_model/'.")
    print("\nКаждый пользователь может использовать собственную модель:")
    print("  1. Поместите файлы вашей модели и токенизатора в папку 'weight_model/'.")
    print(f"  2. Либо скачайте рекомендованную базовую модель {DEFAULT_MODEL}:")
    print(f"     huggingface-cli download {DEFAULT_MODEL} --local-dir weight_model")
    print(f"  3. Либо укажите путь/имя модели через переменную: MODEL_PATH={DEFAULT_MODEL}")
    print("=" * 72)
    sys.exit(1)

print(f"Загрузка токенизатора из {model_source}...")
tokenizer = AutoTokenizer.from_pretrained(model_source)
count = 0
lengths = []

dataset_file = BASE_DIR / "dataset_ml_support.jsonl"
with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text = f"{data['instruction']}\n{data['output']}"
        tokens = tokenizer(text)
        lengths.append(len(tokens["input_ids"]))
        count += 1

print(f"Всего валидных примеров: {count}")
print(f"Минимальная длина токенов: {min(lengths)}")
print(f"Средняя длина токенов: {sum(lengths)//len(lengths)}")
print(f"Максимальная длина токенов: {max(lengths)}")
print(f"Примеров длиннее 512 токенов: {sum(1 for l in lengths if l > 512)}")
