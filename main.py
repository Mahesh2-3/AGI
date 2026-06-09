from openai import OpenAI
from dotenv import load_dotenv
import os

from tools.system_tools import (
    open_notepad,
    create_file,
    list_files
)

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = """
You are a desktop AI assistant.

Available tools:
1. open_notepad()
   Opens Notepad

2. create_file(filename)
   Creates a new file

3. list_files()
   Lists files in current directory

When user asks for an action:
Respond ONLY with this format:

TOOL: tool_name: argument

Examples:
TOOL: open_notepad
TOOL: create_file: notes.txt

If no tool needed, respond normally.
"""

messages=[
    {
        "role":"system",
        "content":SYSTEM_PROMPT
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
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    print(response)



    ai_response = response.choices[0].message.content

    print("AI: ", ai_response)

    if ai_response.startswith("TOOL:"):
        command = ai_response.replace("TOOL:","").strip()
        
        parts = command.split(":")

        tool_name = parts[0].strip()
        argument = None

        if len(parts) > 1 :
            argument = parts[1].strip()

        if tool_name == "open_notepad":
            open_notepad()
        elif tool_name == "create_file":
            create_file(argument)
        elif tool_name == "list_files":
            files = list_files()
            print(f"Files: {files}")


    messages.append({
        "role":"assistant",
        "content":ai_response
    })