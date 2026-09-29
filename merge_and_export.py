import os
import shutil
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = BASE_DIR / "weight_model"
LORA_PATH = BASE_DIR / "lora_output"
MERGED_PATH = BASE_DIR / "merged_model"
ONNX_FP32_PATH = BASE_DIR / "onnx_model_new"



def main():
    print("1. Слияние LoRA адаптера с базовой моделью...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )

    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    merged_model = model.merge_and_unload()

    MERGED_PATH.mkdir(exist_ok=True)
    merged_model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"Слитая модель сохранена в {MERGED_PATH}")

    print("\n2. Экспорт в ONNX FP32...")
    if ONNX_FP32_PATH.exists():
        shutil.rmtree(ONNX_FP32_PATH)
    
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", str(MERGED_PATH),
        "--task", "text-generation-with-past",
        str(ONNX_FP32_PATH)
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
