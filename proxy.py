import os
import time
from pathlib import Path
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_WEIGHTS = BASE_DIR / "weight_model"
MERGED_WEIGHTS = BASE_DIR / "merged_model"

# Определяем источник весов: .env -> merged_model -> weight_model -> HuggingFace default
env_model = os.getenv("MODEL_PATH")
if env_model:
    model_name_or_path = env_model
elif MERGED_WEIGHTS.exists() and any(MERGED_WEIGHTS.iterdir()):
    model_name_or_path = str(MERGED_WEIGHTS)
elif LOCAL_WEIGHTS.exists() and any(LOCAL_WEIGHTS.iterdir()):
    model_name_or_path = str(LOCAL_WEIGHTS)
else:
    model_name_or_path = DEFAULT_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Загрузка модели ({model_name_or_path}) на {device} (тип данных: {dtype})...", flush=True)
start_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map=device
)
print(f"Модель загружена за {time.time() - start_load:.2f} сек.", flush=True)

SYSTEM_PROMPT = "Ты русскоязычный ассистент по машинному обучению. Кратко и строго по делу отвечай на вопросы по ML."


def generate_answer(user_question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - t0

    input_length = inputs["input_ids"].shape[1]
    answer_tokens = outputs[0][input_length:]

    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    if not answer_text:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        answer_tokens = outputs[0][input_length:]
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    print(f"Генерация заняла {gen_time:.2f} сек: {answer_text[:60]}...", flush=True)
    return answer_text


if __name__ == "__main__":
    test_question = "что такое градиентный бустинг?"
    print(f"Вопрос: {test_question}")
    response = generate_answer(test_question)
    print(f"Ответ бота: {response}")