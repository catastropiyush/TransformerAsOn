# Phase Transitions in Large Language Models and the $O(N)$ Model

This repository contains the code for the paper [Phase Transitions in Large Language Models and the $O(N)$ Model](https://arxiv.org/abs/2501.16241).

## File Description

* `gen_beta.py` generates text under diff temperature and stores under `data/`
* `energy_on.py` computes the energy of the text and then plot the energy vs temperature curve

## Prepare Environment

```
python3 -m venv .vllm
source .vllm/bin/activate
pip install -r requirements.txt
```

## Run

```
export model_name="Qwen/Qwen2.5-0.5B"
python3 gen_beta.py
python3 energy_on.py
```
