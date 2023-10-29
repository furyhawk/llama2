# LLM

## Model Zephyr 7B Beta - GGUF

Download the model from here: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF

https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa

https://llama-cpp-python.readthedocs.io/en/latest/

## Install

```sh
# Base ctransformers with no GPU acceleration
pip install ctransformers
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]
# Or with AMD ROCm GPU acceleration (Linux only)
CT_HIPBLAS=1 pip install ctransformers --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems only
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

```sh
pip3 install huggingface-hub
CT_METAL=1 pip install ctransformers --no-binary ctransformers
pip install langchain
pip3 install torch
pip install transformers
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install beautifulsoup4
pip install chromadb
```

## Learning

* default context size for model is 512 tokens. Change it to 2048 tokens.
* `zephyr-7b-beta.Q8_0.gguf` is definately more useful than `zephyr-7b-beta.Q4_K_M.gguf`
