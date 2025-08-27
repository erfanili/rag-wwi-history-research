import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import json
from argparse import ArgumentParser
from utils import parse_args_with_config
from answer_engine import main, Config
with open('src/evaluation/questions_2.json','r') as f:
    data = json.load(f)
    
with open('config.yaml','r') as f:
    entries = yaml.safe_load(f)
        
args, config = parse_args_with_config()



with open(f'src/evaluation/answers/questions_2_{config.retriever}_{config.retriever_topk}_{config.reranker}_{config.reranker_topk}.jsonl','w') as out:
    for item in data:
        question = item['question']
        tag = item['tag']
        chunks,answer, keyword = main(query=question, config = config)

        if chunks is not None:
                    context = "\n\n".join([c['text'] for c,_ in chunks])
        obj = {'tag':tag, 'question': question,'answer':answer}
        out.write(json.dumps(obj)+"\n")