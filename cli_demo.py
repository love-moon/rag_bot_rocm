from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# embedding_model = SentenceTransformer('Dmeta-embedding-zh')
# embedding_model.max_seq_length = 1024
embedding_model = HuggingFaceEmbedding('Dmeta-embedding-zh')
Settings.embed_model = embedding_model
documents = SimpleDirectoryReader("docs").load_data()

index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
# from modelscope.msdatasets import MsDataset
# from modelscope.utils.constant import DownloadMode

# Load the dataset
# ds_train = MsDataset.load('wyj123456/firefly', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
# print(next(iter(ds_train)))


llama_path = "/home/moon/workspace/py/pytorch_learning/Meta-Llama-3.1-8B-Instruct"

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


# messages = [
#     {"role": "user", "content": "写一首诗吧"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, return_tensors="pt"
# ).to(model.device)

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=8192,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True))
