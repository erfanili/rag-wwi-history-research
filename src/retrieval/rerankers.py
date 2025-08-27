
from sentence_transformers import CrossEncoder
from pymilvus import model
import spacy
import os 
import numpy as np



_nlp = None
_cross_encoder = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_cross_encoder(model_name):
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(model_name)

    return _cross_encoder


def rerank_with_cross_encoder(query, docs, topk=30,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

    nlp = get_nlp()
    cross_encoder = get_cross_encoder(model_name)
    sent_doc = []
    paragraphs = [doc["text"] for doc,_ in docs]
    for p in paragraphs:
        sentences = [s.text for s in nlp(p).sents]
        sent_doc.extend(sentences)
    pairs = [(query, s) for s in sent_doc]
    scores = cross_encoder.predict(pairs)  # higher = more relevant
    
    # Sort by score descending
    ranked = sorted(zip(sent_doc, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:topk]
    top_formatted = [({"text":sent},score) for sent,score in top]
    return top_formatted
    


ef = model.dense.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)

def rerank_with_embeds(query, docs, topk=100):
    # Step 1: Extract and split paragraphs into sentences
    nlp = get_nlp()
    paragraphs = [doc["text"] for doc, _ in docs]
    sent_doc = []
    for p in paragraphs:
        sentences = [s.text.strip() for s in nlp(p).sents if s.text.strip()]
        sent_doc.extend(sentences)

    if not sent_doc:
        return []

    # Step 2: Embed query and sentences
    sentence_embeddings = ef.encode_documents(sent_doc)
    query_embedding = ef.encode_queries(query)

    # Step 3: Cosine similarity
    sentence_embeddings = np.array(sentence_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    sentence_embeddings = normalize(sentence_embeddings)
    query_embedding = normalize(query_embedding)

    scores = sentence_embeddings @ query_embedding.T
    scores = scores.flatten()

    # Step 4: Sort and format
    ranked = sorted(zip(sent_doc, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:topk]
    output =[({"text": sent}, float(score)) for sent, score in top]
        
    return output


