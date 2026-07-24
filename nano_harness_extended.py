#!/usr/bin/env python3

import io
import os
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from openai import OpenAI


# ==========================================
# 1. CONFIGURAÇÕES GERAIS
# ==========================================

TASK = """
Use the hf_search tool to search for bert models on Hugging Face.
Summarize the top 3 results returned by the tool.
Do not answer from memory.
"""

MODEL = os.getenv("NANO_MODEL", "zai-org/GLM-5.1")
BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://router.huggingface.co/v1",
)
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN

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
]

DONE = False
FINAL_RESULT = None


# ==========================================
# 2. PROMPT DO SISTEMA
# ==========================================

SYSTEM_PROMPT = f"""
Você é um agente code-first.

Responda somente com código Python executável.
Você pode usar bloco markdown ```python ... ``` ou código puro.

Ferramentas disponíveis:
- list_dir(path=".")
- read_file(path, max_chars=4000)
- write_file(path, content)
- exec_cmd(args)
- final_answer(value)
- print(value)

Regras:
- Não use import.
- Não acesse os, subprocess ou arquivos diretamente.
- Todos os caminhos devem permanecer dentro de: {WORKSPACE}
- Comandos permitidos: {ALLOW_COMMANDS}
- Escrita habilitada: {ALLOW_WRITE}
- Quando terminar, chame obrigatoriamente final_answer("resumo").
- Não repita indefinidamente a mesma ação.
""".strip()


# ==========================================
# 3. FERRAMENTAS
# ==========================================

def safe_path(path="."):
    """Garante que o caminho permaneça dentro do workspace."""
    workspace = WORKSPACE.resolve()
    requested = (workspace / path).resolve()

    if not requested.is_relative_to(workspace):
        raise ValueError(f"O caminho '{path}' sai da pasta do projeto.")

    return requested


def list_dir(path="."):
    """Lista arquivos e pastas."""
    directory = safe_path(path)

    if not directory.is_dir():
        raise NotADirectoryError(f"'{path}' não é uma pasta.")

    return sorted(
        item.name + ("/" if item.is_dir() else "")
        for item in directory.iterdir()
    )


def read_file(path, max_chars=4000):
    """Lê um arquivo de texto com limite de caracteres."""
    file_path = safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"'{path}' não existe.")

    if not file_path.is_file():
        raise IsADirectoryError(f"'{path}' é uma pasta.")

    content = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return content[:max_chars]


def write_file(path, content):
    """Escreve em arquivo somente quando a escrita estiver habilitada."""
    if not ALLOW_WRITE:
        raise PermissionError("A escrita de arquivos está desativada.")

    file_path = safe_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    text = str(content)
    file_path.write_text(text, encoding="utf-8")

    return f"Foram escritos {len(text)} caracteres em '{file_path.name}'."


def exec_cmd(args):
    """Executa somente comandos permitidos."""
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

    return "\n\n".join(output_parts)[:MAX_CHARS]


def final_answer(value):
    """Marca a tarefa como concluída."""
    global DONE, FINAL_RESULT

    DONE = True
    FINAL_RESULT = str(value)

    return FINAL_RESULT


# ==========================================
# 4. EXTRAÇÃO DO CÓDIGO
# ==========================================

def extract_code(content):
    """Aceita bloco python, bloco comum ou código puro."""
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
        raise ValueError("Nenhum código foi encontrado na resposta.")

    return code


# ==========================================
# 5. CHAMADA AO MODELO
# ==========================================

def ask_model(client, messages):
    """Chama o modelo usando Chat Completions."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=1000,
    )

    if not response.choices:
        raise RuntimeError("A API respondeu sem nenhuma opção.")

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


# ==========================================
# 6. LOOP PRINCIPAL
# ==========================================

def main():
    global DONE, FINAL_RESULT

    DONE = False
    FINAL_RESULT = None

    if not HF_TOKEN:
        print("Erro: HF_TOKEN ou OPENAI_API_KEY não foi configurado.")
        return

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=BASE_URL
    )

    print("Cliente configurado.")
    print("Modelo:", MODEL)
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
            print("\nErro ao chamar a API:")
            print(f"{type(error).__name__}: {error}")
            break

        print("\nIA respondeu:")
        print(content if content else "[resposta vazia]")

        if not content:
            print("\nA API respondeu, mas não enviou conteúdo textual.")
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
                "__builtins__": {
                    "str": str,
                    "len": len,
                    "enumerate": enumerate,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "float": float,
                    "int": int,
                    "min": min,
                    "max": max,
                    "sum": sum,
                },
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
                "json": json,
            }

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, exec_globals, {})

            stdout_text = stdout_buffer.getvalue().strip()
            stderr_text = stderr_buffer.getvalue().strip()

            if DONE:
                break

            observations = []

            if stdout_text:
                observations.append(f"stdout:\n{stdout_text}")

            if stderr_text:
                observations.append(f"stderr:\n{stderr_text}")

            result_turn = (
                "\n\n".join(observations)
                or "Código executado com sucesso, sem saída."
            )

        except Exception as error:
            result_turn = (
                "Erro na execução do código: "
                f"{type(error).__name__}: {error}"
            )

        print("\nResultado do Sistema (Observação):")
        print(result_turn)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Resultado da execução anterior:\n"
                    f"{result_turn}\n\n"
                    "Continue a tarefa. Quando terminar, "
                    "chame final_answer()."
                ),
            }
        )

    print("\n==========================================")

    if DONE:
        print("✓ TAREFA CONCLUÍDA COM SUCESSO!")
        print("\nResultado Final:")
        print(FINAL_RESULT)
    else:
        print("✗ A tarefa não foi concluída.")
        print("Consulte as mensagens acima para identificar a causa.")


if __name__ == "__main__":
    main()
