import os
import time
import json
import openai
from openai import RateLimitError, OpenAIError
from dotenv import load_dotenv
import argparse

load_dotenv()
# Set your OpenAI API key via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def grade_answer(question, answer, correct_answer=None):
    """
    Grades the provided answer to the question using OpenAI's LLM.
    Falls back to manual review if LLM fails repeatedly.
    """
    prompt = (
        'You are a accuracy judge for QA. Grade correctness against historical truth. '
        'Ignore style, verbosity, formatting, and reasoning quality unless it introduces factual errors.'
        ' Accept aliases, transliterations, and minor spelling/diacritic variants if they unambiguously refer to the same entity. '
        '- correct (score=2): answers the question correctly and focuses on relevant minor details'
        '- partial (score=1): contains mostly general remarks instead of historical detail or supporting facts'
        '-incorrect (score=0): the answer is wrong or irrelevant to the question or no answer'
        'Try to distinguish correct versus partial'
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Only output the numeric score (0, 1, or 2)."
    )

    if correct_answer:
        prompt += f"\nCorrect answer: {correct_answer}"

    for attempt in range(3):
        try:
            client = openai.OpenAI()

            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
            )

            score_str = resp.choices[0].message.content.strip()
            print(score_str)
            return float(score_str) if score_str in {"0", "1", "2"} else None
        except RateLimitError:
            time.sleep(2 ** attempt)
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            break

    # Fallback for manual review
    return None

def evaluate_batch(input_path, output_path, correct_answers_mapping=None):
    """
    Reads lines of JSON (tag, question, answer), grades each, and writes results.
    """
    correct_answers_mapping = correct_answers_mapping or {}
    counts = {0: 0, 1: 0, 2: 0}
    evaluations = []

    with open(input_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            q = record.get("question")
            a = record.get("answer")
            correct = correct_answers_mapping.get(q)
            score = grade_answer(q, a, correct)
            ticket = {
                "question": q,
                "answer": a,
                "score": score,
                "correct_answer": correct,
            }
            evaluations.append(ticket)
            if score in counts:
                counts[score] += 1

    summary = {
        "counts": counts,
        "accuracy": (counts[1]+counts[2]*2) / (2*len(evaluations)),
        "total": len(evaluations)
    }

    with open(output_path, 'w') as fout:
        json.dump({"summary": summary, "evaluations": evaluations}, fout, indent=2)

    print("Evaluation summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', required=True)
    
    args = parser.parse_args()
    os.makedirs("evaluation/answers",exists_ok=True)
    os.makedirs("evaluation/grades",exist_ok=True)
    evaluate_batch(
        input_path=f"src/evaluation/answers/{args.f}.jsonl",
        output_path=f"src/evaluation/grades/{args.f}_grades.json",
    )
