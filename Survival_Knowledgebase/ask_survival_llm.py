import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# === CONFIG ===
PERSIST_DIR = "index_store"
PDF_FOLDER = "Survival_Knowledgebase"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"  # (or llama2, phi, etc.)

# === SETUP EMBEDDINGS + LLM ===
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = Ollama(model=LLM_MODEL)

# === LOAD OR BUILD INDEX ===
if os.path.exists(PERSIST_DIR):
    print("🔄 Loading saved index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("📥 Reading PDFs and building index...")
    docs = SimpleDirectoryReader(PDF_FOLDER).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Index built and saved.")

# === QUERY LOOP ===
query_engine = index.as_query_engine()
print("🟢 Ask a survival question (type 'exit' to quit):")

while True:
    query = input("\n💬 Your question: ")
    if query.lower() == "exit":
        print("👋 Exiting. Stay safe!")
        break
    response = query_engine.query(query)
    print("\n📘 Answer:\n", response)
