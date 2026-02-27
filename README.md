# Tool Calling and MCP

In the previous lessons, we learned that the LLM is just a text completion engine.
It takes a prompt and returns the most likely next tokens.

In this lesson, we will learn about **tool calling** — one of the most hyped features
in modern AI. By the end of this lesson, you will see that it is just text completion.
Nothing more.

## The Problem

LLMs are trained on static data. They don't know what time it is, they can't search
the web, and they can't look up a Bible verse on demand. They can only complete text.

So how do tools work?

## Tool Calling is Just Text Completion

Let's find out. Here is a prompt that tells Qwen it has access to a tool:

```python
from chat import complete

prompt = """<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city name"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the weather like in Vancouver?<|im_end|>
<|im_start|>assistant
<think>

</think>
"""

print(complete(prompt, max_tokens=200))
```

Run this and you will see something like:

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Vancouver"}}
</tool_call>
```

That's it. The model printed some text. It looks like a function call. It is not a
function call. Nothing was executed. No weather was fetched. The model has no ability
to run code or call APIs.

It just completed the prompt in a way that matches the pattern it was trained on.

## Your Code Does the Work

The tool call JSON is just text sitting in a string. It's up to your code to decide
what to do with it. Here is the simplest possible dispatch loop:

```python
import json
import re
from chat import complete

def get_weather(city):
    # Pretend we called a weather API
    return f"It is 12°C and raining in {city}."

def run(user_input):
    prompt = f"""<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "get_weather", "description": "Get the current weather for a city", "parameters": {{"type": "object", "properties": {{"city": {{"type": "string", "description": "The city name"}}}}, "required": ["city"]}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""

    response = complete(prompt, max_tokens=200)
    print("Raw model output:", response)

    # Parse the tool call out of the response text
    match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    if match:
        tool_call = json.loads(match.group(1).strip())
        name = tool_call['name']
        args = tool_call['arguments']

        # Dispatch: your code decides what to do
        if name == 'get_weather':
            result = get_weather(**args)
            print("Tool result:", result)
        else:
            print(f"Unknown tool: {name}")
    else:
        # No tool call, just a regular response
        print("Response:", response)

run("What's the weather like in Vancouver?")
```

Notice what the dispatch loop is doing:

1. It calls `complete()` and gets back a string
2. It searches that string for `<tool_call>` tags using a regular expression
3. It parses the JSON inside those tags
4. It checks the function name and calls the right Python function

The model didn't call anything. Your `if name == 'get_weather'` statement called
the function. You are in complete control. You can ignore the tool call, log it,
ask for user confirmation, or call something else entirely.

## Exercise 1

Modify the dispatch loop above to add a second tool called `get_time` that returns
the current time. Register it in the tools list in the prompt, add a branch to the
dispatch loop, and test it with a question like "What time is it?".

Observe what happens when you ask a question that doesn't need a tool, like "What
is 2 + 2?". Does the model always try to call a tool?

## What is MCP?

MCP stands for Model Context Protocol. It was announced by Anthropic in late 2024
and generated a lot of excitement in the AI community.

Here is what MCP actually is:

1. A standardized JSON format for describing tools
2. A standardized JSON format for the model to request a tool call
3. A convention for hosting tools as a server so other applications can use them

That's it. You just built the core of it in the exercise above.

The value of MCP is not in the protocol itself — it's in the standardization. If
everyone agrees on the same JSON format, then tools built by one person can be used
by any application that speaks MCP. It's the same reason HTTP is useful: not because
it's clever, but because everyone agreed to use it.

> ⚠️ Instructor Note:
> Students may have seen breathless coverage of MCP as a major AI breakthrough.
> This is a good moment to discuss the difference between a useful standard and
> a genuinely new capability. MCP is the former. The underlying mechanism —
> structured text completion followed by code dispatch — is unchanged.

## Building a Bible RAG MCP Server

Now let's build something real. We will expose our Bible RAG search from Lesson 4
as an MCP tool, served over FastAPI.

This means any application that speaks MCP can now search the Bible by asking
our server for the `search_bible` tool.

Here is the server in `mcp_server.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import rag

app = FastAPI(title="Bible RAG MCP Server")

# MCP tool definition - this is what you advertise to clients
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_bible",
            "description": "Search the King James Bible for verses relevant to a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to search for"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of verses to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

class ToolCallRequest(BaseModel):
    name: str
    arguments: dict

@app.get("/tools")
def list_tools():
    """Advertise available tools to MCP clients"""
    return TOOLS

@app.post("/call")
def call_tool(request: ToolCallRequest):
    """Execute a tool call and return the result"""
    if request.name == "search_bible":
        query = request.arguments["query"]
        top_k = request.arguments.get("top_k", 3)
        verses = rag.search(query, top_k)
        return {"result": verses}
    else:
        return {"error": f"Unknown tool: {request.name}"}
```

Run the server with:

```
uvicorn mcp_server:app --reload
```

Now here is a client in `mcp_client.py` that uses this server as a tool source for
the LLM:

```python
import requests
import json
import re
from chat import complete

# Fetch tool definitions from the MCP server
tools = requests.get("http://localhost:8000/tools").json()

# Build the tools section of the prompt dynamically from server response
tools_str = '\n'.join(json.dumps(t) for t in tools)

def chat(user_input):
    prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions about the Bible.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""

    response = complete(prompt, max_tokens=300)

    # Check if the model wants to call a tool
    match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    if match:
        tool_call = json.loads(match.group(1).strip())

        # Send the tool call to the MCP server
        result = requests.post("http://localhost:8000/call", json=tool_call).json()
        context = '\n'.join(result['result'])

        # Now call the model again with the tool result injected as context
        followup = f"""<|im_start|>system
Use the following Bible verses to answer the user's question:
{context}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
        return complete(followup, max_tokens=300)
    else:
        return response

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    print("AI:", chat(user_input))
```

Try running the client and asking questions like "What does the Bible say about
forgiveness?" or "Tell me about David and Goliath". Watch how the model decides
to call the search tool, your code sends the request to the server, and the result
comes back as context for the final answer.

Compare this to the RAG chatbot from Lesson 4. In Lesson 4, the search was
hardcoded into the main loop — it ran on every message whether it was needed or not.
Here, the model decides when to search. That is the only meaningful difference
MCP introduces.

## Exercise 2

The current client always does a second LLM call after the tool result. Modify it
so that if the model responds without a tool call, it just returns the response
directly without a second call.

## Exercise 3

Add a second tool to the MCP server called `get_book_info` that takes a book name
(e.g. "Genesis") and returns a short description of that book. You can hardcode
a dictionary of descriptions. Add it to the tools list in the server and add the
dispatch branch in the `/call` endpoint. Verify the client picks it up automatically
from `/tools` without any changes to the client code.

This is the point of the standardized protocol — the client doesn't need to know
about new tools in advance.

## What You Built

Across these five lessons, you built the entire stack from scratch:

- **Lesson 1**: Context is just text you inject into a prompt
- **Lesson 2**: The model is just completing tokens based on probability
- **Lesson 3**: Meaning can be encoded as vectors and searched by similarity
- **Lesson 4**: RAG automates context injection using vector search
- **Lesson 5**: Tool calling is structured text completion your code interprets, and MCP is just an agreed-upon format for doing that across applications

Every AI feature you will ever build reduces to one of these patterns. The frameworks
and libraries hide them, but they do not change them.

# Submit

When you are ready to submit, execute the following commands in the terminal:

```bash
$ git add -A
$ git commit -m 'submit'
$ git push
```