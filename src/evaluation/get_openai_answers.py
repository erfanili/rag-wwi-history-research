import openai
import json
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()  # uses OPENAI_API_KEY from environment

def answer_question_with_openai(question: str, model="gpt-4o") -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in WWI history. Answer in 1 to 3 paragraphs."},
        {"role": "user", "content": question}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error for question: {question}\n{e}")
        return None

with open('src/evaluation/Q.json','r') as f:
    questions =json.load(f)

results = []

with open("src/evaluation/openai_answers.jsonl","w") as f:

    for q in questions:
        print(f"Answering: {q['question']}")
        answer = answer_question_with_openai(q["question"])
        f.write(json.dumps({
            "question": q["question"],
            "answer": answer
        })+"\n")
        sleep(1.5)  # avoid rate limit

# Save results


print("Saved to openai_answers.json")
