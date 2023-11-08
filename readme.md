# LLM


## Model Card for Zephyr 7B β

<img src="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png" alt="Zephyr Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>


Download the model from here:

* https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
* https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF

Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-β is the second model in the series, and is a fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) that was trained on on a mix of publicly available, synthetic datasets using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290). We found that removing the in-built alignment of these datasets boosted performance on [MT Bench](https://huggingface.co/spaces/lmsys/mt-bench) and made the model more helpful. However, this means that model is likely to generate problematic text when prompted to do so and should only be used for educational and research purposes. You can find more details in the [technical report](https://arxiv.org/abs/2310.16944).


## Performance

At the time of release, Zephyr-7B-β is the highest ranked 7B chat model on the [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmarks:

| Model | Size | Alignment | MT-Bench (score) | AlpacaEval (win rate %) |
|-------------|-----|----|---------------|--------------|
| StableLM-Tuned-α | 7B| dSFT |2.75| -|
| MPT-Chat |  7B |dSFT |5.42| -|
| Xwin-LMv0.1 | 7B| dPPO| 6.19| 87.83|
| Mistral-Instructv0.1 | 7B|  - | 6.84 |-|
| Zephyr-7b-α |7B|  dDPO| 6.88| -|
| **Zephyr-7b-β** 🪁 | **7B** | **dDPO** | **7.34** | **90.60** |
| Falcon-Instruct |  40B |dSFT |5.17 |45.71|
| Guanaco | 65B |  SFT |6.41| 71.80|
| Llama2-Chat |  70B |RLHF |6.86| 92.66|
| Vicuna v1.3 |  33B |dSFT |7.12 |88.99|
| WizardLM v1.0 |  70B |dSFT |7.71 |-|
| Xwin-LM v0.1 |   70B |dPPO |- |95.57|
| GPT-3.5-turbo | - |RLHF |7.94 |89.37|
| Claude 2 |  - |RLHF |8.06| 91.36|
| GPT-4 |  -| RLHF |8.99| 95.28|

In particular, on several categories of MT-Bench, Zephyr-7B-β has strong performance compared to larger open models like Llama2-Chat-70B:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6200d0a443eb0913fa2df7cc/raxvt5ma16d7T23my34WC.png)

However, on more complex tasks like coding and mathematics, Zephyr-7B-β lags behind proprietary models and more research is needed to close the gap.

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
pip install lark
```

```sh
pip install -U openai-whisper
brew install ffmpeg
pip install torchaudio
pip install scipy
pip install ipywidgets
pip install matplotlib
```
## Learning

* default context size for model is 512 tokens. Change it to 2048 tokens.
* `zephyr-7b-beta.Q8_0.gguf` is definately more useful than `zephyr-7b-beta.Q4_K_M.gguf`

## Roadblocks

* Deep coupling of `langchain` and `openai` api. `create_conversational_retrieval_agent` for example does not work with `llama-cpp-python`. `SelfQueryRetriever.from_llm` encounter JSON decoding error.