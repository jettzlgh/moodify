import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import disk_offload , init_empty_weights, infer_auto_device_map


def get_transformers_lyrics():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # with init_empty_weights():
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={"cpu": "16GB", "gpu": "8GB", "disk": "100GB"}  # Adjust limits to your setup
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    # disk_offload(model=model, offload_dir="offload")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
