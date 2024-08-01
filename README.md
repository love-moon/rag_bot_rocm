# GETTING START
## SYSTEM REQUIREMENTS
Ubuntu 22.04

ROCM 6.1 on AMD graph cards

git and git-lfs

Python 3.9 or later

cmake

Latest Pytorch on Rocm 6.1
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

## LLMs REQUIREMENTS
Meta-Llama-3.1-8B-Instruct

git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

Dmeta-embedding
git clone https://huggingface.co/DMetaSoul/Dmeta-embedding-zh

like
```bash
├── Dmeta-embedding-zh
├── Meta-Llama-3.1-8B-Instruct
├── cli_demo.py
└── docs
```

## PYTHON PACKAGE REQUIREMENTS

### llama_index install
```bash
pip3 install llama_index llama-index-embeddings-huggingface llama-index-llms-huggingface
```
### bitsandbytes installation
see
https://huggingface.co/docs/bitsandbytes/main/en/installation#amd-gpu


## START
put some text file into folder 'docs', then run

`python3 cli_demo.py`