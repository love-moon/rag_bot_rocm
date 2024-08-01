from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# init Embedding
embedding_model = HuggingFaceEmbedding('Dmeta-embedding-zh')
Settings.embed_model = embedding_model

# load user documents
documents = SimpleDirectoryReader("docs").load_data()

# create index
index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)



llama_path = "./Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path)


Settings.llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, device_map="0")

query_engine = index.as_query_engine()
# response = query_engine.query("What did the beast ask from the beauty?")
# print(response)
while True:
    query = input("\nUser:")
    if len(query) == 0:
        continue

    if query == "exit":
        exit(0)
    
    print(query_engine.query(query))


