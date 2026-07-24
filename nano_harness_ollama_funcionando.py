#!/usr/bin/env python3

import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from openai import OpenAI


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

TASK = os.getenv(
    "NANO_TASK",
    "Search for bert models on Hugging Face and summarize top 3.",
)

MODEL = os.getenv("NANO_MODEL", "qwen2.5-coder:3b")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
HF_TOKEN = os.getenv("HF_TOKEN", "")

WORKSPACE = Path.cwd()
MAX_STEPS = 8
TIMEOUT_S = 30
MAX_CHARS = 8000
ALLOW_WRITE = False
TEMPERATURE = 0.2

ALLOW_COMMANDS = [
    "ls",
    "cat",
    "pwd",
    "echo",
    "head",
    "tail",
    "wc",
    "rg",
    "find",
    "git",
    "grep",
]

DONE = False
FINAL_RESULT = None


# ==========================================================
# PROMPT DO AGENTE
# ==========================================================

SYSTEM_PROMPT = f"""
You are a code-first agent.

Reply only with executable Python code, preferably inside a
```python
...
```
block.

Available tools:
- list_dir(path='.')
- read_file(path, max_chars=4000)
- write_file(path, content)
- exec_cmd(args)
- web_fetch(url, max_bytes=10000)
- hf_search(query, resource_type='models', limit=5)
- git_log(limit=10)
- json_parse(json_string)
- compute_stats(numbers)
- final_answer(value)
- print(value)

Rules:
- Do not use import.
- Do not access os, subprocess, urllib, or files directly.
- All paths must remain inside: {WORKSPACE}
- Allowed commands: {ALLOW_COMMANDS}
- Writes enabled: {ALLOW_WRITE}
- Always use print() when you need a tool result in the next step.
- Avoid repeating the same tool call.
- Finish by calling final_answer(result).
""".strip()


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def clip(value, limit=MAX_CHARS):
    text = str(value)

    if len(text) <= limit:
        return text

    return text[:limit] + "\n...[truncated]"


def safe_path(path="."):
    workspace = WORKSPACE.resolve()
    requested = (workspace / path).resolve()

    if not requested.is_relative_to(workspace):
        raise ValueError(f"O caminho '{path}' sai da pasta do projeto.")

    return requested


# ==========================================================
# TOOLS
# ==========================================================

def list_dir(path="."):
    directory = safe_path(path)

    if not directory.is_dir():
        raise NotADirectoryError(f"'{path}' não é uma pasta.")

    return sorted(
        item.name + ("/" if item.is_dir() else "")
        for item in directory.iterdir()
    )


def read_file(path, max_chars=4000):
    file_path = safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"'{path}' não existe.")

    if not file_path.is_file():
        raise IsADirectoryError(f"'{path}' é uma pasta.")

    content = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return clip(content, min(max_chars, MAX_CHARS))


def write_file(path, content):
    if not ALLOW_WRITE:
        raise PermissionError("write_file está desativado.")

    file_path = safe_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    text = str(content)
    file_path.write_text(text, encoding="utf-8")

    return f"Foram escritos {len(text)} caracteres."


def exec_cmd(args):
    if not isinstance(args, list):
        raise TypeError("Os argumentos precisam estar em uma lista.")

    if not args:
        raise ValueError("Nenhum comando foi informado.")

    command = args[0]

    if command not in ALLOW_COMMANDS:
        raise PermissionError(f"O comando '{command}' não é permitido.")

    result = subprocess.run(
        args,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )

    output_parts = []

    if result.stdout:
        output_parts.append(f"stdout:\n{result.stdout}")

    if result.stderr:
        output_parts.append(f"stderr:\n{result.stderr}")

    if not output_parts:
        output_parts.append(
            f"Comando finalizado com código {result.returncode}."
        )

    return clip("\n\n".join(output_parts), MAX_CHARS)


def web_fetch(url, max_bytes=10000):
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "nano-harness/1.0"},
        )

        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT_S,
        ) as response:
            content = response.read(max_bytes + 1)

        if len(content) > max_bytes:
            content = content[:max_bytes] + b"\n...[truncated]"

        return content.decode("utf-8", errors="replace")

    except urllib.error.URLError as error:
        return f"Error: Failed to fetch {url}: {error}"

    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"


def hf_search(query, resource_type="models", limit=5):
    try:
        safe_limit = max(1, min(int(limit), 20))

        params = urllib.parse.urlencode(
            {
                "search": query,
                "limit": safe_limit,
            }
        )

        url = f"https://huggingface.co/api/{resource_type}?{params}"

        headers = {"User-Agent": "nano-harness/1.0"}

        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        request = urllib.request.Request(
            url,
            headers=headers,
        )

        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT_S,
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        results = []

        for item in data[:safe_limit]:
            results.append(
                {
                    "id": item.get("id"),
                    "downloads": item.get("downloads", 0),
                    "description": (
                        item.get("description", "") or ""
                    )[:200],
                }
            )

        return results

    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"


def git_log(limit=10):
    try:
        safe_limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        return "Error: limit precisa ser um número inteiro."

    return exec_cmd(
        [
            "git",
            "log",
            "--oneline",
            f"-{safe_limit}",
        ]
    )


def json_parse(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as error:
        return f"Error: {error}"


def compute_stats(numbers):
    try:
        nums = [float(number) for number in numbers]

        if not nums:
            return "Error: a lista está vazia."

        return {
            "min": min(nums),
            "max": max(nums),
            "mean": sum(nums) / len(nums),
            "count": len(nums),
        }

    except (TypeError, ValueError) as error:
        return f"Error: valores inválidos: {error}"


def final_answer(value):
    global DONE, FINAL_RESULT

    DONE = True
    FINAL_RESULT = value

    return value


# ==========================================================
# EXECUÇÃO DO CÓDIGO GERADO PELO MODELO
# ==========================================================

def extract_code(content):
    if not content:
        raise ValueError("A IA retornou uma resposta vazia.")

    code_match = re.search(
        r"```(?:python)?\s*(.*?)```",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if code_match:
        code = code_match.group(1).strip()
    else:
        code = content.strip()

    if not code:
        raise ValueError("Nenhum código foi encontrado.")

    return code


def ask_model(client, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=1000,
    )

    if not response.choices:
        raise RuntimeError("A API respondeu sem opções.")

    content = response.choices[0].message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return ""


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

def main():
    global DONE, FINAL_RESULT

    DONE = False
    FINAL_RESULT = None

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=BASE_URL,
    )

    print("Cliente configurado.")
    print("Modelo:", MODEL)
    print("Endpoint:", BASE_URL)
    print("Workspace:", WORKSPACE)
    print("Tarefa:", TASK)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": TASK,
        },
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n[Passo {step}/{MAX_STEPS}] Chamando a IA...")

        try:
            content = ask_model(client, messages)

        except Exception as error:
            print("\nErro ao chamar o modelo:")
            print(f"{type(error).__name__}: {error}")
            break

        print("\nIA respondeu:")
        print(content if content else "[resposta vazia]")

        if not content:
            print("\nO modelo respondeu sem texto.")
            break

        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

        try:
            code = extract_code(content)

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            exec_globals = {
                "__builtins__": {},
                "list_dir": list_dir,
                "read_file": read_file,
                "write_file": write_file,
                "exec_cmd": exec_cmd,
                "web_fetch": web_fetch,
                "hf_search": hf_search,
                "git_log": git_log,
                "json_parse": json_parse,
                "compute_stats": compute_stats,
                "final_answer": final_answer,
                "print": print,
                "json": json,
            }

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(
                    code,
                    exec_globals,
                    {},
                )

            stdout_text = stdout_buffer.getvalue().strip()
            stderr_text = stderr_buffer.getvalue().strip()

            if DONE:
                break

            observations = []

            if stdout_text:
                observations.append(
                    f"stdout:\n{clip(stdout_text)}"
                )

            if stderr_text:
                observations.append(
                    f"stderr:\n{clip(stderr_text)}"
                )

            result_turn = (
                "\n\n".join(observations)
                or "Executado com sucesso, sem saída."
            )

        except FileNotFoundError:
            result_turn = (
                "Error: FileNotFoundError: arquivo não encontrado."
            )

        except PermissionError as error:
            result_turn = f"Error: PermissionError: {error}"

        except subprocess.TimeoutExpired:
            result_turn = (
                "Error: TimeoutError: comando demorou demais."
            )

        except Exception as error:
            result_turn = f"Error: {type(error).__name__}: {error}"

        print("\nResultado do Sistema (Observação):")
        print(result_turn)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Resultado da execução anterior:\n"
                    f"{result_turn}\n\n"
                    "Continue. Quando terminar, chame final_answer()."
                ),
            }
        )

    print("\n==========================================")

    if DONE:
        print("✓ TAREFA CONCLUÍDA COM SUCESSO!")
        print("\nResultado Final:")
        print(clip(FINAL_RESULT, MAX_CHARS))
    else:
        print("✗ A tarefa não foi concluída.")


if __name__ == "__main__":
    main()
