import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    load_index_from_storage,
    StorageContextS
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# === Step 1: Load PDFs ===
PDF_FOLDER = "Survival_Knowledgebase"  # Update to your actual folder name
PERSIST_DIR = "index_storage"
documents = SimpleDirectoryReader(PDF_FOLDER).load_data()

# === Step 2: Set up CallbackManager and LLM ===
debug_handler = LlamaDebugHandler()  # Optional: shows debug info
callback_manager = CallbackManager([debug_handler])  # Can be empty if you want

llm = Ollama(model="phi3", callback_manager=callback_manager)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# === Step 3: Load or Build Index ===
if os.path.exists(PERSIST_DIR):
    print("Loading existing index")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Index not found. Building new index")
    documents = SimpleDirectoryReader(PDF_FOLDER).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("index built")

# === Step 4: Query loop ===
query_engine = index.as_query_engine(similarity_top_k=5)
print("Ask a question (type 'exit' to quit):")
while True:
    question = input("\n💬 Your question: ")
    if question.lower() == "exit":
        break
    try:
        response = query_engine.query(question)
        print("\n📘 Answer:\n", response)
    except Exception as e:
        print("❌ Error:", e)
