from openai import OpenAI
import os
import json
import urllib.request
import urllib.error
import pandas as pd
import gradio as gr
import requests

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MODEL = os.getenv("NANO_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
MAX_STEPS = 10


# --- Tool Definitions (OpenAI function calling format) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's content. Use this for attached files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return the final answer to the question. Call this when you have the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


# --- Tool Implementations ---
def web_search(query):
    """Search the web using DDGS."""
    try:
        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=5,
        )

        formatted_results = []

        for item in results:
            title = item.get("title", "").strip()
            url = item.get("href", "").strip()
            snippet = item.get("body", "").strip()

            if not url:
                continue

            formatted_results.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {snippet}"
            )

        if not formatted_results:
            return "No results found."

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


def web_fetch(url, max_bytes=100000):
    """Fetch webpage content and extract readable text."""
    try:
        from bs4 import BeautifulSoup

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read(max_bytes).decode("utf-8", errors="replace")

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
            tag.decompose()

        main = (
            soup.select_one("#mw-content-text")
            or soup.select_one(".mw-parser-output")
            or soup.select_one("main")
            or soup.select_one("article")
            or soup.body
            or soup
        )

        text = main.get_text(separator="\n")

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        clean_text = "\n".join(lines)

        return clean_text[:20000] if clean_text else "No readable text found."

    except Exception as e:
        return f"Fetch error: {type(e).__name__}: {e}"

def read_file(path):
    """Read a file's content."""
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(10000)
    except Exception as e:
        return f"File error: {e}"


def calculator(expression):
    """Evaluate a math expression safely."""
    import math
    allowed = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "len": len})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


def execute_tool(name, args):
    """Dispatch tool calls."""
    if name == "web_search":
        return web_search(args.get("query", ""))
    elif name == "web_fetch":
        return web_fetch(args.get("url", ""))
    elif name == "read_file":
        return read_file(args.get("path", ""))
    elif name == "calculator":
        return calculator(args.get("expression", ""))
    elif name == "final_answer":
        return None
    return "Unknown tool."


# --- Agent ---
class ToolAgent:
    def __init__(self):
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY nao configurado.")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print(f"ToolAgent initialized. Model: {MODEL}")

    def __call__(self, question: str) -> str:
        print(f"\nQuestion: {question[:100]}...")

        messages = [
            {
    "role": "system",
    "content": (
        "You are an autonomous research agent.\n\n"
        "Your goal is to answer questions correctly, not quickly.\n\n"
        "Rules:\n"
        "- Never guess.\n"
        "- If you don't know, search the web.\n"
        "- If search returns useful URLs, fetch the page.\n"
        "- Read the fetched content before answering.\n"
        "- If the first search fails, try a different search query.\n"
        "- Use calculator for mathematical questions.\n"
        "- Use read_file when a question refers to an attached file.\n"
        "- Only call final_answer when you are confident.\n"
        "- Return only the final answer, with no explanations or markdown."
    ),
},
            {"role": "user", "content": question},
        ]

        for step in range(MAX_STEPS):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                max_tokens=500,
            )

            msg = response.choices[0].message

            if msg.content:
                messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls or []})
            elif msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            else:
                messages.append({"role": "assistant", "content": msg.content or ""})

            if not msg.tool_calls:
                return (msg.content or "").strip()

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"  Tool: {fn_name}({fn_args})")

                if fn_name == "final_answer":
                    return fn_args.get("answer", "").strip()

                result = execute_tool(fn_name, fn_args)
                print(f"  Result: {str(result)[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return "Max steps reached without answer."


# --- Run & Submit ---
def run_and_submit_all():
    username = "Jariny"
    api_url = DEFAULT_API_URL

    try:
        agent = ToolAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = "https://github.com/JaryStefano/nano-harness-pratica/tree/main/Final_Assignment_Template"

    # Fetch questions
    try:
        resp = requests.get(f"{api_url}/questions", timeout=15)
        resp.raise_for_status()
        questions_data = resp.json()
        if not questions_data:
            return "No questions received.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # Run agent
    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"ERROR: {e}"})

    if not answers_payload:
        return "No answers produced.", pd.DataFrame(results_log)

    # Submit
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    try:
        resp = requests.post(f"{api_url}/submit", json=submission_data, timeout=60)
        resp.raise_for_status()
        result_data = resp.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', '')}"
        )
    except requests.exceptions.HTTPError as e:
        error_detail = f"Status {e.response.status_code}."
        try:
            error_detail += f" {e.response.json().get('detail', '')}"
        except Exception:
            pass
        final_status = f"Submission Failed: {error_detail}"
    except Exception as e:
        final_status = f"Submission Failed: {e}"

    return final_status, pd.DataFrame(results_log)


# --- Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Runner")
    gr.Markdown("Click below to run the agent on all questions and submit.")

    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Results", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    print("Launching...")
    demo.launch(debug=True, share=False)
