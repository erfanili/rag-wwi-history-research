#answer_engine.py
import argparse
import subprocess
import yaml
import sys,os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from src.retrieval.retrievers import zillis,splade,sparse_retrieve
from src.retrieval.rerankers import rerank_with_cross_encoder, rerank_with_embeds
from utils import parse_args_with_config
import os
from dotenv import load_dotenv

#utils.py

import spacy
import requests
from together import Together
import os
import json
import numpy as np
from pymilvus import model, MilvusClient
import torch
from sentence_transformers import SparseEncoder, CrossEncoder
from dotenv import load_dotenv
from huggingface_hub import login
from utils import load_config, Config


load_dotenv(override=True)



def run_together(prompt,model: str = "llama3.1:8b-instruct"):
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    if model == "llama3.1:8b-instruct":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    return response.choices[0].message.content


def run_ollama(prompt: str, model: str = "llama3.1:8b-instruct") -> str:
    # if model == "llama3.1:8b-instruct":
    model_name = "llama3.1:8b-instruct-q8_0"
    # model_name = model
    """Send prompt to Ollama and return decoded output."""
    # subprocess.run(["ollama", "pull", model_name], check=True)
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        check=True
    )
    return result.stdout.decode()

def build_prompt(chunks, question):
    
    
    if chunks is None:
        prompt = f"""You are a historian specializing in World War I. Answer the following question based on historical facts. Only give the answer.
        Answer the question below.
        Avoid general statements.
        Try to keep all relevant details in your answer.
        Avoid trivial statements.
        Answer between 3 and 5 paragraphs.
        Only give the answer.
        

        
        Question: {question}
        
        Answer:
        """
        
        return prompt
        
    else:
        context = " ".join([c["text"] for c,_ in chunks])
        # prompt = f"""You are a historian specializing in World War I. Answer the following question based on historical facts. Only give the answer.
        
        prompt = f"""You are a historian specializing in World War I.
        You are given a question and a context. 
        Step 1: find the parts of the context that are relevant to the question.
        Step 2: put the relevant parts into a coherent long passage
        Step 2: use the passage to answer the question definitively with all relevant details in the context.
        Answer the  question **strictly based on the provided context**.
        Don't say "Based on the provided context" or similar expressions.
        Avoid general statements.
        Try to keep all relevant details in your answer.
        Avoid trivial statements.
        Answer between 3 and 5 paragraphs.
        Only give the answer.
        

        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        return prompt

import subprocess

def expand_query(query):
    prompt = f"""
You are a query rewriter for a retrieval system.
Rewrite vague or underspecified queries into expanded, precise questions that
make the query unambiguous. Also give a keyword that best represents the visual or symbolic core of the query,
suitable for searching images on Wikimedia Commons or similar platforms.

Here are some examples:

Example 1:
Query: What happened in 1915?
Expanded Query: What happened in 1915 during World War I?
Keyword: World War I 1915

Example 2:
Query: What happened during summer 1915?
Expanded Query: What were the major events during summer 1915 in World War I?
Keyword: World War I summer 1915

Example 3:
Query: Who was Haig?
Expanded Query: Who was Haig during World War I? What were major facts about him in the war?
Keyword: Douglas Haig

Example 4:
Query: What is Gallipoli?
Expanded Query: What was the Gallipoli campaign during World War I?
Keyword: Gallipoli campaign

Example 5:
Query: Why did the US join the war?
Expanded Query: What were the reasons that led the United States to enter World War I?
Keyword: US entry World War I

Example 6:
Query: Outcome of Verdun?
Expanded Query: What was the outcome of the Battle of Verdun, including casualties and morale?
Keyword: Battle of Verdun

Example 7:
Query: What countries won the war?
Expanded Query: Which countries were victorious at the end of World War I? Name them.
Keyword: World War I victors

Example 8:
Query: German casualties in 1916
Expanded Query: What was the number of German casualties in the war during the year 1916? Give numbers.
Keyword: German army 1916

Example 9:
Query: When did the war begin?
Expanded Query: What was the exact date when World War I began?
Keyword: World War I outbreak

Example 10:
Query: Who was British military commander?
Expanded Query: Who was the British military commander in World War I? Name them.
Keyword: British generals World War I

Now give the Expanded Query and Keyword.

Format your response like this:
Expanded Query: <your rewritten query>
Keyword: <1–3 keywords for image search>

Only emit these two lines. Do not say anything else.

Query: {query}
Expanded Query:
"""
    # result = subprocess.run(["ollama", "run", "llama3:instruct"], input=prompt.encode(), stdout=subprocess.PIPE)
    # output = result.stdout.decode()
    
    output = run_together(prompt)

    lines = output.strip().splitlines()
    expanded_query = ""
    keyword = ""
    for line in lines:
        if line.startswith("Expanded Query:"):
            expanded_query = line.replace("Expanded Query:", "").strip()
        elif line.startswith("Keyword:"):
            keyword = line.replace("Keyword:", "").strip()

    return expanded_query, keyword



def main(query,config):
    """Build prompt from query+chunks, run LLM, and return output."""
    
    retriever_topk = config.retriever_topk
    reranker_topk = config.reranker_topk
    if config.expand_query:
        expanded_query, keyword = expand_query(query)

    else:
        expanded_query = query
        keyword = 'World War I'

    if config.retriever == "":
        prompt = build_prompt(chunks=None, question=expanded_query)
        print('No retrieval. Answering without context.')
        top_chunks = None
        
    else:
        
        if config.retriever == 'splade':
            top_chunks = splade(query=query, 
                                chunks_path=config.chunks_relative_path, 
                                doc_embs_path=config.splade_embds_relative_path,
                                topk=retriever_topk)
        elif config.retriever == 'openai_embeds':
            top_chunks = zillis(expanded_query,topk=retriever_topk)
        else:
            raise ValueError("No Valid Retriever.")
        
        
        if config.rerank:
            # breakpoint()
            if config.reranker == 'cross_encoder':
                
                top_chunks = rerank_with_cross_encoder(query=query, docs=top_chunks, topk = reranker_topk)
                
            elif config.reranker == 'openai_embeds':
                top_chunks = rerank_with_embeds(query=query,docs=top_chunks,topk=reranker_topk)

            else:
                print('No Valid Reranking Model. Skipping Reranking.')
                

        prompt = build_prompt(chunks=top_chunks, question=expanded_query)
    

    if config.run == "ollama":
        output = run_ollama(prompt,model=config.llm_model)
    elif config.run == "together":
        output = run_together(prompt, model=config.llm_model)
    
    else:
        raise ValueError("No Valid LLM runenr specified (ollama | together).")
    return top_chunks, output, keyword








if __name__ == "__main__":
    try:
        args, config = parse_args_with_config()
        query = getattr(args, "q", "").strip()
        
        if not query:
            print("Error: No question provided. Use the --q argument.")
            exit(1)
        
        _, output, keyword = main(query=query, config=config)

        print("Output:\n", output)
        print("Keyword(s):\n", keyword)

    except Exception as e:
        print(f"Execution failed: {e}")
        exit(1)