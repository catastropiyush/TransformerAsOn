#! /usr/bin/env python3

import os
import re
import math
import json
import numpy as np
from matplotlib import pyplot as plt

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = os.getenv("model_name")
print("using %s"%(model_name))

if not model_name[0].isalpha():
    model_name = os.path.expanduser(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
embed_tokens = model.get_input_embeddings()

def get_files(dir, file_perfix):
    text_files = []
    for i in os.listdir("data/%s"%(dir)):
        r = re.search("%s-T([0-9\.]+?)\.json"%(file_perfix), i)
        if not r:
            continue
        T = float(r.group(1))
        text_files.append({
            "T": T,
            "path": "data/%s/%s"%(dir,i)
        })
    text_files.sort(key=lambda x: x["T"])
    return text_files

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data

def add_log(filename, entry, data):
    json_data = load_json(filename)
    if entry not in json_data:
        json_data[entry] = data
    else:
        json_data[entry] = {**json_data[entry], **data}

    lines = []
    for k, v in json_data.items():
        lines.append("\"%s\": %s"%(k, json.dumps(v)))
    with open(filename, "w") as f:
        # json.dump(json_data, f, indent=4)
        f.write("{\n")
        f.write(",\n".join(lines))
        f.write("\n}")

def energy(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    with torch.no_grad():
        energies = []
        lengths  = []
        for d in data:
            model_inputs  = tokenizer(d["generated"], return_tensors="pt").to(model.device)
            inputs_embeds = embed_tokens(model_inputs["input_ids"]) # 1, len, 1536
            inputs_embeds = inputs_embeds.squeeze(0)
            inputs_embeds/= torch.linalg.vector_norm(inputs_embeds, dim = -1, keepdim = True)
            attn = -inputs_embeds@inputs_embeds.T # attn[i,j] = ei cdot ej
            attn.fill_diagonal_(0.0)
            e = (attn.sum()/attn.shape[0]).item()
            energies.append(e)
            lengths.append(attn.shape[0])
    print("avg length for %s: %.2f"%(filename, np.mean(lengths)))
    return np.mean(energies), np.std(energies)/math.sqrt(len(data))

def susceptibility(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    N = 512
    with torch.no_grad():
        spins = []
        for d in data:
            model_inputs  = tokenizer(d["generated"], return_tensors="pt").to(model.device)
            inputs_embeds = embed_tokens(model_inputs["input_ids"]) # 1, len, 1536
            inputs_embeds = inputs_embeds.squeeze(0)
            inputs_embeds/= torch.linalg.vector_norm(inputs_embeds, dim = -1, keepdim = True)
            spins.append(inputs_embeds[-N:,:]) # N x 896

    spins = torch.stack(spins)
    spinsbar = spins.mean(dim=0)
    deltaspins = spins - spinsbar # 103 x 1000 x 896
    sisj = []
    for i in range(N):
        for j in range(i+1,N):
            sisj.append((deltaspins[:,i,:]*deltaspins[:,j,:]).sum(dim=-1).mean().item())
    return np.mean(sisj)

def plot_energy(plot_flag=True):
    dir, file_perfix = model_name.split("/")[-2:] # Qwen, Qwen2.5-32B
    text_files = get_files(dir, file_perfix)

    betas, energies, sigmas = [], [], []
    for i in text_files:
        T, path = i["T"], i["path"]
        e, s = energy(path)
        betas.append(T)
        energies.append(e)
        sigmas.append(s)

    print(list(zip(betas, [float("%2.f"%(e)) for e in energies])))
    add_log("data/%s-energy.json"%(dir), file_perfix, {
        "model_size": model.num_parameters(exclude_embeddings=True),
        "betas": betas,
        "energies": energies,
        "sigmas": sigmas
    })

    if not plot_flag:
        return

    Tmin = round(min(betas))
    Tmax = round(max(betas))

    plt.errorbar(betas, energies, yerr=sigmas, capsize=2)
    plt.minorticks_on()
    plt.grid(True, which="both")
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.title(file_perfix)
    plt.tight_layout()
    plt.savefig("energy_%s-%s-%s.pdf"%(file_perfix,Tmin,Tmax))
    print("saved")

if __name__=="__main__":
    plot_energy()
