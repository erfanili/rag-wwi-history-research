from pymilvus import MilvusClient,model
import os
from sentence_transformers import SparseEncoder
import torch

import json
import re
from data_processing.huggingface_data import download_file
from huggingface_hub import login
from rank_bm25 import BM25Okapi



def zillis(query,topk=100):
    COLLECTION = "wwi_history"

    ef = model.dense.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    client = MilvusClient(
        uri = os.getenv("MILVUS_URI"),
        token = os.getenv("MILVUS_TOKEN")
    )


    embedding = ef.encode_queries([query])

    search_res = client.search(
        collection_name=COLLECTION,
        data=embedding,  # Use the `emb_text` function to convert the question to an embedding vector
        limit=topk,  # Return top 3 results
        search_params={"metric_type": "COSINE"},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    return[(item["entity"],item["distance"]) for item in  search_res[0]]





def splade(query,chunks_path,doc_embs_path,topk=5):
    download_file(chunks_path)
    with open(os.path.join("downloaded_data",chunks_path), "r") as f:
        chunks = [json.loads(line) for line in f]
    download_file(doc_embs_path)
    doc_embs = torch.load(os.path.join("downloaded_data",doc_embs_path), map_location=torch.device("cpu"))

    login(token=os.getenv("HF_API_KEY"))
    model = SparseEncoder("naver/splade-v3").to("cpu")
    queries = [query]
    query_embeddings = model.encode_query(queries)
    scores = model.similarity(query_embeddings, doc_embs).squeeze()
    top_scores, top_indices =torch.topk(scores,k=topk)
    
    return [(chunks[i], float(top_scores[j])) for j,i in enumerate(top_indices)]



def sparse_retrieve(query,chunks,topk=5):
    
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    
    tokenized_corpus = [tokenize(doc["text"]) for doc in chunks]
    tokenized_q = tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_q)
    top_ids =scores.argsort()[::-1][:topk]
    
    return [(chunks[i], scores[i]) for i in top_ids]

