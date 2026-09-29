import shutil
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MERGED_PATH = BASE_DIR / "merged_model"
ONNX_FP32_PATH = BASE_DIR / "onnx_model_new"

print(f"1. Экспорт слитой модели из {MERGED_PATH} в ONNX...", flush=True)
if ONNX_FP32_PATH.exists():
    shutil.rmtree(ONNX_FP32_PATH)
ONNX_FP32_PATH.mkdir(exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, local_files_only=True)

ort_model = ORTModelForCausalLM.from_pretrained(
    MERGED_PATH,
    export=True,
    use_cache=True,
    local_files_only=True
)

ort_model.save_pretrained(ONNX_FP32_PATH)
tokenizer.save_pretrained(ONNX_FP32_PATH)
print(f"ONNX FP32 модель сохранена в {ONNX_FP32_PATH}", flush=True)