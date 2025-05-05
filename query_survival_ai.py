import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PERSIST_DIR = "index_store"

# Set local embedding model
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = None  # ✅ Disable OpenAI LLM so you can run offline

# Check for cached index
if os.path.exists(PERSIST_DIR):
    print("🔄 Loading existing index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("📥 Loading documents from Survival_Knowledgebase...")
    docs = SimpleDirectoryReader("Survival_Knowledgebase").load_data()
    print(f"✅ Loaded {len(docs)} document(s).")

    print("⚙️ Building index and saving to disk...")
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Index built and persisted.")

# Query engine
query_engine = index.as_query_engine()
print("🟢 Ready! Ask your survival question (or type 'exit' to quit):")

while True:
    query = input("\n💬 Your question: ")
    if query.lower() == "exit":
        print("👋 Exiting. Stay safe out there!")
        break
    response = query_engine.query(query)
    print("\n📘 Answer:\n", response)
