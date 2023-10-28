# LLM


https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF

https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa

https://llama-cpp-python.readthedocs.io/en/latest/

```sh
pip3 install huggingface-hub
CT_METAL=1 pip install ctransformers --no-binary ctransformers
pip install langchain
pip3 install torch
pip install transformers
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```