#! /usr/bin/env python3

import os
import json
import numpy as np
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import torch
import torch.distributed as dist
from datasets import load_dataset
from vllm import LLM, SamplingParams

# export HF_DATASETS_OFFLINE=1
# ds = load_dataset("sentence-transformers/simple-wiki")
ds = load_dataset("parquet", data_files="/home/youran/.cache/huggingface/datasets/sentence-transformers__simple-wiki/pair/train-00000-of-00001.parquet")

# export model_name="Qwen/Qwen2.5-0.5B"
# export model_name="Qwen/Qwen2.5-1.5B"
# export model_name="Qwen/Qwen2.5-3B"
# export model_name="Qwen/Qwen2.5-7B"
# export model_name="Qwen/Qwen2.5-14B"
# export model_name="Qwen/Qwen2.5-32B"
model_name = os.getenv("model_name")
print("using %s"%(model_name))

if not model_name[0].isalpha():
    model_name = os.path.expanduser(model_name)

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len = 2048,
    # tensor_parallel_size=torch.cuda.device_count(),
    # dtype="float16"
)

def gen_with_T(T, max_tokens = 1024):
    T = float("%.4f"%(T))
    params = SamplingParams(
        n = 1,
        temperature = T,
        max_tokens = max_tokens,
        min_tokens = max_tokens-1,
        seed = 42,
        skip_special_tokens = True
    )

    model_family  = model_name.split("/")[-2]
    file_perfix = model_name.split("/")[-1]
    fname = "data/%s/%s-T%s.json"%(model_family, file_perfix, T)

    if not os.path.exists("./data/%s"%(model_family)):
        os.makedirs("./data/%s"%(model_family), exist_ok=True)
        print("created ./data/%s"%(model_family))

    if os.path.exists(fname):
        print("%s already exists, skipping"%(fname))
        return

    try:
        outputs = llm.generate([i for i in ds['train']['text']][::1000], params)
        data = []
        for output in outputs:
            prompt = output.prompt
            generated = output.outputs[0]
            data.append({
                "prompt": prompt,
                "generated": generated.text
            })
        with open(fname, "w") as f:
            json.dump(data, f, indent = 4, ensure_ascii=False)
        print("saved to %s"%(fname))
    finally:
        # or there will be some warning from torch
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__=="__main__":
    for T in np.linspace(0,10,11):
        gen_with_T(T)

    for T in np.linspace(0,2,21):
        gen_with_T(T)
