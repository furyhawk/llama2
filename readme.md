# LLM

```sh
pip3 install huggingface-hub
CT_METAL=1 pip install ctransformers --no-binary ctransformers
pip install langchain
pip3 install torch
pip install transformers
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```