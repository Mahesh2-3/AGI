from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

messages=[
    {
        "role":"system",
        "content":"You are a helpful Ai Assistant."
    }
]

while True:
    user = input("You: ")

    if("clear" in user or "exit" in user):
        break

    messages.append({
        "role":"user",
        "content":user
    })

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )


    ai_response = response.choices[0].message.content

    print("AI: ", ai_response)

    messages.append({
        "role":"assistant",
        "content":ai_response
    })