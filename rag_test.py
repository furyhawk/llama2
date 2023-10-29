# %%
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# %%
llama_model_path = "../../models/zephyr-7b-beta.Q4_K_M.gguf"
#Use Llama model for embedding
embeddings = LlamaCppEmbeddings(model_path=llama_model_path, n_ctx=2048) # , n_ctx=2048

# %%
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=llama_model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

# %%
loader = TextLoader("../../datasets/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# %%
texts

# %%

# embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)



# %%
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


