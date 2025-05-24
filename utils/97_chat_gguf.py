# pip install gguf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "janhq/Qwen3-14B-v0.1-deepresearch-100-step-gguf"
filename = "temp_model-q4_k_m.gguf"

torch_dtype = torch.float32 # could be torch.float16 or torch.bfloat16 too
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, torch_dtype=torch_dtype)

# ask em an com chua
print(tokenizer.decode(model.generate(tokenizer.encode("Hello, em an com chua?"), max_length=100)[0]))