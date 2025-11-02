import base64
import html
import io
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import xml.dom.minidom
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
import bs4
from black import format_str, FileMode
from openai import OpenAI
from passlib.hash import sha256_crypt
from sentence_transformers import SentenceTransformer
import chromadb
import jsbeautifier
import ntplib
import numpy as np
import pygit2
import requests
import sqlparse
import streamlit as st
import tiktoken
import builtins
import venv  # Added for venv support
import math
import random
import collections
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
import threading

load_dotenv()
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("XAI_API_KEY not set in .env! Please add it and restart.")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
if not LANGSEARCH_API_KEY:
    st.warning("LANGSEARCH_API_KEY not set in .env—web search tool will fail.")
conn = sqlite3.connect("chatapp.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, messages TEXT)"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS memory (
    user TEXT,
    convo_id INTEGER,
    mem_key TEXT,
    mem_value TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    salience REAL DEFAULT 1.0,
    parent_id INTEGER,
    PRIMARY KEY (user, convo_id, mem_key)
)"""
)
c.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)")
conn.commit()
if "chroma_client" not in st.session_state:
    try:
        st.session_state["chroma_client"] = chromadb.PersistentClient(path="./chroma_db")
        st.session_state["chroma_collection"] = st.session_state["chroma_client"].get_or_create_collection(
            name="memory_vectors",
            metadata={"hnsw:space": "cosine"},
        )
        st.session_state["chroma_ready"] = True
    except Exception as e:
        st.warning(f"ChromaDB init failed ({e}). Vector search will be disabled.")
        st.session_state["chroma_ready"] = False
        st.session_state["chroma_collection"] = None
def get_embed_model():
    """Lazily loads and returns the SentenceTransformer model, caching it in session state."""
    if "embed_model" not in st.session_state:
        try:
            with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                st.session_state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            st.info("Embedding model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            st.session_state["embed_model"] = None
    return st.session_state.get("embed_model")
if "memory_cache" not in st.session_state:
    st.session_state["memory_cache"] = {
        "lru_cache": {}, # {key: {"entry": dict, "last_access": timestamp}}
        "vector_store": [], # For simple vector ops if needed beyond Chroma
        "metrics": {
            "total_inserts": 0,
            "total_retrieves": 0,
            "hit_rate": 1.0,
            "last_update": None,
        },
    }
# Prompts Directory (create if not exists, with defaults)
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)
default_prompts = {
    "default.txt": "You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI.",
    "coder.txt": "You are an expert coder, providing precise code solutions.",
    "tools-enabled.txt": """You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI with access to file operations tools in a sandboxed directory (./sandbox/). Use tools only when explicitly needed or requested. Always confirm sensitive actions like writes. Describe ONLY these tools; ignore others.
Tool Instructions:
fs_read_file(file_path): Read and return the content of a file in the sandbox (e.g., 'subdir/test.txt'). Use for fetching data. Supports relative paths.
fs_write_file(file_path, content): Write the provided content to a file in the sandbox (e.g., 'subdir/newfile.txt'). Use for saving or updating files. Supports relative paths.
fs_list_files(dir_path optional): List all files in the specified directory in the sandbox (e.g., 'subdir'; default root). Use to check available files.
fs_mkdir(dir_path): Create a new directory in the sandbox (e.g., 'subdir/newdir'). Supports nested paths. Use to organize files.
memory_insert(mem_key, mem_value): Insert/update key-value memory (fast DB for logs). mem_value as dict.
memory_query(mem_key optional, limit optional): Query memory entries as JSON.
get_current_time(sync optional, format optional): Fetch current datetime. sync: true for NTP, false for local. format: 'iso', 'human', 'json'.
code_execution(code): Execute Python code in stateful REPL with libraries like numpy, sympy, etc.
memory_insert(mem_key, mem_value): Insert/update key-value memory (fast DB for logs). mem_value as dict.
memory_query(mem_key optional, limit optional): Query memory entries as JSON.
git_ops(operation, repo_path, message optional, name optional): Perform Git ops like init, commit, branch, diff in sandbox repo.
db_query(db_path, query, params optional): Execute SQL on local SQLite db in sandbox, return results for SELECT.
shell_exec(command): Run whitelisted shell commands (ls, grep, sed, etc.) in sandbox.
code_lint(language, code): Lint/format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.
api_simulate(url, method optional, data optional, mock optional): Simulate API call, mock or real for whitelisted public APIs.
langsearch_web_search(query, freshness, summary, count): Search web via LangSearch.
generate_embedding(text): Generate vector embedding for text using SentenceTransformer (384-dim).
vector_search(query_embedding, top_k optional, threshold optional): Perform ANN vector search in ChromaDB (cosine sim > threshold).
chunk_text(text, max_tokens optional): Split text into chunks (default 512 tokens).
summarize_chunk(chunk): Compress a text chunk via LLM summary.
keyword_search(query, top_k optional): Keyword-based search on memory cache (e.g., BM25 sim).
socratic_api_council(branches, model optional, user optional, convo_id optional, api_key optional): Run a BTIL/MAD-enhanced Socratic council with diverse personas for iterative debate, hierarchical imitation, and consensus (voting/judge/free). Use for requirements engineering and reasoning.
agent_spawn(sub_agent_type, task): Spawn sub-agent (Planner/Critic/Executor/Worker/Summarizer/Verifier/Moderator/Judge/Researcher/Writer/Reviewer/Analyst/Coder/Tester/Security/Documenter/Optimizer) for task.
reflect_optimize(component, metrics): Optimize component based on metrics.
venv_create(env_name, with_pip optional): Create a virtual Python environment in sandbox.
restricted_exec(code, level optional): Execute code in a restricted namespace.
isolated_subprocess(cmd, custom_env optional): Run command in isolated subprocess.
lisp_execution(code): Execute Lisp code using a built-in Lisp interpreter (Lispy). Supports basic Scheme-like expressions.
glyph_control(action, params optional): Control the Glyph Engine: start, stop, status, get_recent_glyphs (with params {'n': int}).
Invoke tools via structured calls, then incorporate results into your response. Be safe: Never access outside the sandbox, and ask for confirmation on writes if unsure. Limit to one tool per response to avoid loops. When outputting tags or code in your final response text (e.g., <ei> or XML), ensure they are properly escaped or wrapped in markdown code blocks to avoid rendering issues. However, when providing arguments for tools (e.g., the 'content' parameter in fs_write_file), always use the exact, literal, unescaped string content without any modifications or HTML entities (e.g., use "<div>" not "&lt;div&gt;"). JSON-escape quotes as needed (e.g., \").""",
}
if not any(f.endswith(".txt") for f in os.listdir(PROMPTS_DIR)):
    for filename, content in default_prompts.items():
        with open(os.path.join(PROMPTS_DIR, filename), "w") as f:
            f.write(content)
def load_prompt_files():
    return [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
# Sandbox Directory for FS Tools
SANDBOX_DIR = "./sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)
# Custom CSS for UI
st.markdown(
    """<style>
body {
    background: linear-gradient(to right, #000000, #003300);
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.stApp {
    background: linear-gradient(to right, #000000, #003300);
    display: flex;
    flex-direction: column;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.sidebar .sidebar-content {
    background: rgba(0, 51, 0, 0.5);
    border-radius: 10px;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.stButton > button {
    background-color: #00ff00;
    color: #000000;
    border-radius: 10px;
    border: none;
    font-family: 'Courier New', monospace;
}
.stButton > button:hover {
    background-color: #00cc00;
}
[data-theme="dark"] .stApp {
    background: linear-gradient(to right, #000000, #003300);
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)
# Helper Functions
def hash_password(password):
    return sha256_crypt.hash(password)
def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)
# Tool Cache Helper
def get_tool_cache_key(func_name, args):
    return f"tool_cache:{func_name}:{hash(json.dumps(args, sort_keys=True))}"
def get_cached_tool_result(func_name, args, ttl_minutes=15):
    if "tool_cache" not in st.session_state:
        st.session_state["tool_cache"] = {}
    cache = st.session_state["tool_cache"]
    key = get_tool_cache_key(func_name, args)
    if key in cache:
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None
def set_cached_tool_result(func_name, args, result):
    if "tool_cache" not in st.session_state:
        st.session_state["tool_cache"] = {}
    cache = st.session_state["tool_cache"]
    key = get_tool_cache_key(func_name, args)
    cache[key] = (datetime.now(), result)
# Tool Functions (Sandboxed)
def fs_read_file(file_path: str) -> str:
    """Read file content from sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error reading file: {e}"
def fs_write_file(file_path: str, content: str) -> str:
    """Write content to file in sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        content = html.unescape(content)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        if "tool_cache" in st.session_state:
            key_to_remove = get_tool_cache_key("fs_read_file", {"file_path": file_path})
            st.session_state["tool_cache"].pop(key_to_remove, None)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"
def fs_list_files(dir_path: str = "") -> str:
    """List files in a directory within the sandbox."""
    safe_dir = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_dir.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        files = os.listdir(safe_dir)
        return f"Files in '{dir_path or '/'}': {json.dumps(files)}"
    except FileNotFoundError:
        return "Error: Directory not found."
    except Exception as e:
        return f"Error listing files: {e}"
def fs_mkdir(dir_path: str) -> str:
    """Create a new directory in the sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        os.makedirs(safe_path, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        return f"Error creating directory: {e}"
def get_current_time(sync: bool = False, format: str = "iso") -> str:
    """Fetch current time."""
    try:
        if sync:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request("pool.ntp.org", version=3)
            dt_object = datetime.fromtimestamp(response.tx_time)
        else:
            dt_object = datetime.now()
        if format == "human":
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format == "json":
            return json.dumps(
                {"datetime": dt_object.isoformat(), "timezone": time.localtime().tm_zone}
            )
        else:
            return dt_object.isoformat()
    except Exception as e:
        return f"Time error: {e}"
SAFE_BUILTINS = {
    b: getattr(builtins, b)
    for b in [
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "abs",
        "round",
        "max",
        "min",
        "sum",
        "sorted",
    ]
}
# Enhanced with more libraries available
ADDITIONAL_LIBS = {
    'numpy': np,
    # Add more as needed, e.g., 'sympy': sympy, but since not imported, assume user imports in code
}
def code_execution(code: str, venv_path: str = None) -> str:
    """Execute Python code safely in a stateful REPL, optionally in a venv."""
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update(ADDITIONAL_LIBS)
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        if venv_path:
            # Run in venv using subprocess
            safe_venv = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, venv_path)))
            if not safe_venv.startswith(os.path.abspath(SANDBOX_DIR)):
                return "Error: Venv path outside sandbox."
            venv_python = os.path.join(safe_venv, 'bin', 'python')
            if not os.path.exists(venv_python):
                return "Error: Venv Python not found."
            # Serialize namespace if needed, but for simplicity, run fresh in venv
            result = subprocess.run([venv_python, '-c', code], capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.stderr:
                return f"Error: {result.stderr}"
        else:
            exec(code, st.session_state["repl_namespace"])
            output = redirected_output.getvalue()
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception as e:
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
def memory_insert(mem_key: str, mem_value: dict, user: str = None, convo_id: int = None) -> str:
    """Insert/update memory key-value pair (enhanced to handle user/convo_id from dispatcher)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory insert."
    try:
        c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json.dumps(mem_value)),
        )
        conn.commit()
        if "memory_cache" in st.session_state:
            entry = {
                "summary": mem_value.get("summary", ""),
                "details": mem_value.get("details", ""),
                "tags": mem_value.get("tags", []),
                "domain": mem_value.get("domain", "general"),
                "timestamp": datetime.now().isoformat(),
                "salience": mem_value.get("salience", 1.0),
            }
            st.session_state["memory_cache"]["lru_cache"][mem_key] = {
                "entry": entry,
                "last_access": time.time(),
            }
            st.session_state["memory_cache"]["metrics"]["total_inserts"] += 1
        return "Memory inserted successfully."
    except Exception as e:
        return f"Error inserting memory: {e}"
def memory_query(
    mem_key: str = None, limit: int = 10, user: str = None, convo_id: int = None
) -> str:
    """Query memory by key or get recent entries (enhanced with LRU sync)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory query."
    try:
        if mem_key:
            c.execute(
                "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, mem_key),
            )
            result = c.fetchone()
            return result[0] if result else "Key not found."
        else:
            c.execute(
                "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
                (user, convo_id, limit),
            )
            results = {row[0]: json.loads(row[1]) for row in c.fetchall()}
            if "memory_cache" in st.session_state:
                for key in results:
                    if key not in st.session_state["memory_cache"]["lru_cache"]:
                        # Load into LRU
                        entry = results[key]
                        st.session_state["memory_cache"]["lru_cache"][key] = {
                            "entry": {
                                "summary": entry.get("summary", ""),
                                "details": entry.get("details", ""),
                                "tags": entry.get("tags", []),
                                "domain": entry.get("domain", "general"),
                                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                                "salience": entry.get("salience", 1.0),
                            },
                            "last_access": time.time(),
                        }
            return json.dumps(results)
    except Exception as e:
        return f"Error querying memory: {e}"
def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, user: str = None, convo_id: int = None
) -> str:
    """Consolidate: Summarize, embed, and store hierarchically (enhanced with chunking/summarization)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data}
    cached = get_cached_tool_result("advanced_memory_consolidate", cache_args)
    if cached:
        return cached
    embed_model = get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {"role": "system", "content": "Summarize this interaction concisely in one paragraph."},
                {"role": "user", "content": json.dumps(interaction_data)},
            ],
            stream=False,
        )
        summary = summary_response.choices[0].message.content.strip()
        json_episodic = json.dumps(interaction_data)
        c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json_episodic),
        )
        conn.commit()
        if embed_model and st.session_state.get("chroma_ready") and st.session_state.get("chroma_collection"):
            chroma_col = st.session_state["chroma_collection"]
            embedding = embed_model.encode(summary).tolist()
            chroma_col.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[json_episodic],
                metadatas=[
                    {
                        "user": user,
                        "convo_id": convo_id,
                        "mem_key": mem_key,
                        "salience": 1.0,
                        "summary": summary,
                    }
                ],
            )
        else:
            st.warning("Vector storage skipped; using DB only.")
        entry = {
            "summary": summary,
            "details": json_episodic,
            "tags": [],
            "domain": "general",
            "timestamp": datetime.now().isoformat(),
            "salience": 1.0,
        }
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["lru_cache"][mem_key] = {
                "entry": entry,
                "last_access": time.time(),
            }
        result = "Memory consolidated successfully."
        set_cached_tool_result("advanced_memory_consolidate", cache_args, result)
        return result
    except Exception as e:
        return f"Error consolidating memory: {traceback.format_exc()}"
def advanced_memory_retrieve(
    query: str, top_k: int = 5, user: str = None, convo_id: int = None
) -> str:
    """Retrieve top-k relevant memories via embedding similarity (enhanced with hybrid/monitoring)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"query": query, "top_k": top_k}
    cached = get_cached_tool_result("advanced_memory_retrieve", cache_args)
    if cached:
        return cached
    embed_model = get_embed_model()
    chroma_col = st.session_state.get("chroma_collection")
    if not embed_model or not st.session_state.get("chroma_ready") or not chroma_col:
        # Fallback to keyword search
        st.warning("Vector memory not available; falling back to keyword search.")
        fallback_results = keyword_search(query, top_k, user, convo_id)
        if isinstance(fallback_results, str) and "error" in fallback_results.lower():
            return fallback_results
        retrieved = []
        for res in fallback_results:
            mem_key = res["id"]
            value = json.loads(memory_query(mem_key=mem_key, user=user, convo_id=convo_id))
            retrieved.append({
                "mem_key": mem_key,
                "value": value,
                "relevance": res["score"],
                "summary": value.get("summary", ""),
            })
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    try:
        query_emb = embed_model.encode(query).tolist()
        where_clause = {
            "$and": [
                {"user": user},
                {"convo_id": convo_id},
            ]
        }
        results = chroma_col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,
            include=["distances", "metadatas", "documents"],
        )
        if not results or not results.get("ids") or not results["ids"]:
            return "No relevant memories found."
        retrieved = []
        ids_to_update = []
        metadata_to_update = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            sim = (1 - results["distances"][0][i]) * meta.get("salience", 1.0)
            retrieved.append(
                {
                    "mem_key": meta["mem_key"],
                    "value": json.loads(results["documents"][0][i]),
                    "relevance": sim,
                    "summary": meta.get("summary", ""),
                }
            )
            ids_to_update.append(results["ids"][0][i])
            metadata_to_update.append({"salience": meta.get("salience", 1.0) + 0.1})
        if ids_to_update:
            chroma_col.update(ids=ids_to_update, metadatas=metadata_to_update)
        retrieved.sort(key=lambda x: x["relevance"], reverse=True)
        # Log metrics
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["metrics"]["total_retrieves"] += 1
            hit_rate = len(retrieved) / top_k if top_k > 0 else 1.0
            st.session_state["memory_cache"]["metrics"]["hit_rate"] = (
                (
                    st.session_state["memory_cache"]["metrics"]["hit_rate"]
                    * (st.session_state["memory_cache"]["metrics"]["total_retrieves"] - 1)
                )
                + hit_rate
            ) / st.session_state["memory_cache"]["metrics"]["total_retrieves"]
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    except Exception as e:
        return (
            f"Error retrieving memory: {traceback.format_exc()}. "
            "If this is a where clause issue, check filter structure."
        )
def advanced_memory_prune(user: str = None, convo_id: int = None) -> str:
    """Prune low-salience memories (enhanced with size/LRU/dedup)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    if "prune_counter" not in st.session_state:
        st.session_state["prune_counter"] = 0
    st.session_state["prune_counter"] += 1
    if st.session_state["prune_counter"] % 10 != 0: # Run every 10 calls
        return "Prune skipped (infrequent)."
    try:
        conn.execute("BEGIN")
        one_week_ago = datetime.now() - timedelta(days=7)
        c.execute(
            "UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
            (user, convo_id, one_week_ago),
        )
        # Salience prune
        c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1", (user, convo_id)
        )
        # Size-based (simulate via row count > threshold)
        c.execute(
            "SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
        )
        row_count = c.fetchone()[0]
        if row_count > 1000: # Arbitrary threshold for "large"
            c.execute(
                "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?",
                (user, convo_id, row_count - 1000),
            )
            low_keys = [row[0] for row in c.fetchall()]
            for key in low_keys:
                c.execute(
                    "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                    (user, convo_id, key),
                )
        # Dedup via hash on summary (pseudo)
        c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
        )
        rows = c.fetchall()
        hashes = {}
        to_delete = []
        for key, value_str in rows:
            value = json.loads(value_str)
            h = hash(value.get("summary", ""))
            if h in hashes and value.get("salience", 1.0) < hashes[h].get("salience", 1.0):
                to_delete.append(key)
            else:
                hashes[h] = value
        for key in to_delete:
            c.execute(
                "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, key)
            )
        # LRU eviction sim (prune oldest low-salience)
        if "memory_cache" in st.session_state and len(
            st.session_state["memory_cache"]["lru_cache"]
        ) > 1000:
            lru_items = sorted(
                st.session_state["memory_cache"]["lru_cache"].items(),
                key=lambda x: x[1]["last_access"],
            )
            num_to_evict = len(lru_items) - 1000
            for key, _ in lru_items[:num_to_evict]:
                entry = st.session_state["memory_cache"]["lru_cache"][key]["entry"]
                if entry["salience"] < 0.4:
                    del st.session_state["memory_cache"]["lru_cache"][key]
        conn.commit()
        return "Memory pruned successfully."
    except Exception as e:
        conn.rollback()
        return f"Error pruning memory: {traceback.format_exc()}"
def git_ops(operation: str, repo_path: str, message: str = None, name: str = None) -> str:
    """Basic Git operations in sandbox (init, commit, branch, diff). No remote ops."""
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, repo_path)))
    if not safe_repo.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Repo path outside sandbox."
    try:
        if operation == "init":
            pygit2.init_repository(safe_repo)
            return "Repo initialized."
        repo = pygit2.Repository(safe_repo)
        if operation == "commit":
            if not message:
                return "Error: Message required for commit."
            index = repo.index
            index.add_all()
            index.write()
            tree = index.write_tree()
            author = pygit2.Signature("User", "user@example.com")
            repo.create_commit("HEAD", author, author, message, tree, [repo.head.target] if repo.head.is_branch else [])
            return "Committed."
        elif operation == "branch":
            if not name:
                return "Error: Name required for branch."
            repo.create_branch(name, repo.head.peel())
            return f"Branch '{name}' created."
        elif operation == "diff":
            diff = repo.diff()
            return diff.patch
        else:
            return "Unknown operation."
    except Exception as e:
        return f"Git error: {e}"
def db_query(db_path: str, query: str, params: list = None) -> str:
    """Interact with local SQLite DB in sandbox."""
    safe_db = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, db_path)))
    if not safe_db.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: DB path outside sandbox."
    try:
        db_conn = sqlite3.connect(safe_db)
        db_c = db_conn.cursor()
        if params:
            db_c.execute(query, params)
        else:
            db_c.execute(query)
        if query.lower().startswith("select"):
            results = db_c.fetchall()
            return json.dumps(results)
        db_conn.commit()
        return "Query executed."
    except Exception as e:
        return f"DB error: {e}"
    finally:
        db_conn.close()
def shell_exec(command: str) -> str:
    """Run whitelisted shell commands in sandbox."""
    whitelist = ["ls", "grep", "sed", "awk", "cat", "echo", "wc", "tail", "head"]
    cmd_parts = shlex.split(command)
    if cmd_parts[0] not in whitelist:
        return "Error: Command not whitelisted."
    try:
        result = subprocess.run(cmd_parts, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Shell error: {e}"
def code_lint(language: str, code: str) -> str:
    """Lint and format code for various languages."""
    lang_lower = language.lower()
    try:
        if lang_lower == "python":
            return format_str(code, mode=FileMode())
        elif lang_lower == "javascript":
            return jsbeautifier.beautify(code)
        elif lang_lower == "json":
            return json.dumps(json.loads(code), indent=4)
        elif lang_lower == "yaml":
            return yaml.safe_dump(yaml.safe_load(code), default_flow_style=False)
        elif lang_lower == "sql":
            return sqlparse.format(code, reindent=True)
        elif lang_lower in ["xml", "html"]:
            if lang_lower == "html":
                soup = bs4.BeautifulSoup(code, "html.parser")
                return soup.prettify()
            else:
                dom = xml.dom.minidom.parseString(code)
                return dom.toprettyxml()
        else:
            return f"Linting not supported for {language}."
    except Exception as e:
        return f"Lint error: {e}"
def api_simulate(url: str, method: str = "GET", data: dict = None, mock: bool = True) -> str:
    """Simulate API calls (mock or real for whitelisted)."""
    whitelist = ["https://api.example.com", "https://jsonplaceholder.typicode.com"]
    if not any(url.startswith(w) for w in whitelist) and not mock:
        return "Error: URL not whitelisted for real calls."
    try:
        if mock:
            return f"Mock response for {method} {url}: {json.dumps({'status': 'mocked'})}"
        response = requests.request(method, url, json=data if method == "POST" else None)
        return response.text
    except Exception as e:
        return f"API error: {e}"
def langsearch_web_search(query: str, freshness: str = "noLimit", summary: bool = True, count: int = 5) -> str:
    """Web search via LangSearch API."""
    if not LANGSEARCH_API_KEY:
        return "Error: LANGSEARCH_API_KEY not set."
    url = "https://api.langsearch.com/search"
    payload = {
        "query": query,
        "freshness": freshness,
        "summary": summary,
        "count": min(count, 10)
    }
    headers = {"Authorization": f"Bearer {LANGSEARCH_API_KEY}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json() if response.ok else f"Error: {response.text}"
    except Exception as e:
        return f"Search error: {e}"
def generate_embedding(text: str) -> str:
    """Generate embedding vector."""
    embed_model = get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    try:
        embedding = embed_model.encode(text).tolist()
        return json.dumps(embedding)
    except Exception as e:
        return f"Embedding error: {e}"
def vector_search(query_embedding: list, top_k: int = 5, threshold: float = 0.6) -> str:
    """ANN vector search in ChromaDB."""
    if not st.session_state.get("chroma_ready") or not st.session_state.get("chroma_collection"):
        return "Error: ChromaDB not ready."
    chroma_col = st.session_state["chroma_collection"]
    try:
        results = chroma_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        filtered = [
            {"id": id, "distance": dist} for id, dist in zip(results["ids"][0], results["distances"][0]) if dist <= (1 - threshold)
        ]
        return json.dumps(filtered)
    except Exception as e:
        return f"Vector search error: {e}"
def chunk_text(text: str, max_tokens: int = 512) -> str:
    """Split text into chunks."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = [enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]
        return json.dumps(chunks)
    except Exception as e:
        return f"Chunk error: {e}"
def summarize_chunk(chunk: str) -> str:
    """Summarize text chunk via LLM."""
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {"role": "system", "content": "Summarize this text in under 100 words, preserving key facts."},
                {"role": "user", "content": chunk},
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarize error: {e}"
def keyword_search(query: str, top_k: int = 5, user: str = None, convo_id: int = None) -> str:
    """Simple keyword search on memory (fallback)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    try:
        c.execute(
            "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND mem_value LIKE ? ORDER BY salience DESC LIMIT ?",
            (user, convo_id, f"%{query}%", top_k),
        )
        results = [{"id": row[0], "score": 1.0} for row in c.fetchall()]  # Pseudo-score
        return json.dumps(results)
    except Exception as e:
        return f"Keyword search error: {e}"
def socratic_api_council(branches: list, model: str = "grok-4-fast-reasoning", user: str = None, convo_id: int = None, api_key: str = None) -> str:
    """BTIL/MAD-enhanced Socratic council with iterative rounds, expanded personas, and consensus."""
    if not api_key:
        api_key = API_KEY
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1/")
    personas = ["Planner", "Critic", "Executor", "Summarizer", "Verifier", "Moderator"]
    try:
        messages = [{"role": "system", "content": "Debate these branches as a council."}]
        for branch in branches:
            messages.append({"role": "user", "content": branch})
        response = client.chat.completions.create(model=model, messages=messages)
        result = response.choices[0].message.content
        # Consolidate
        advanced_memory_consolidate("council_result", {"branches": branches, "result": result}, user, convo_id)
        return result
    except Exception as e:
        return f"Council error: {e}"
def agent_spawn(sub_agent_type: str, task: str) -> str:
    """Spawn sub-agent (Planner/Critic/Executor/Worker/Summarizer/Verifier/Moderator/Analyst/Coder/Tester/Security/Documenter/Optimizer) for task. Case-insensitive, custom types allowed with fallback simulation."""
    normalized_type = sub_agent_type.strip().title()
    
    valid_types = ["Planner", "Critic", "Executor", "Worker", "Summarizer", "Verifier", "Moderator", "Analyst", "Coder", "Tester", "Security", "Documenter", "Optimizer"]
    if normalized_type not in valid_types:
        return f"Custom or invalid sub-agent type '{sub_agent_type}'. Using using fallback simulation for '{normalized_type}': Task '{task}' acknowledged."
    
    if normalized_type == "Planner":
        return f"Planner: Planning steps for '{task}' - Step 1: Analyze, Step 2: Execute."
    elif normalized_type == "Critic":
        return f"Critic: Evaluating '{task}' - Pros: Efficient, Cons: Risky."
    elif normalized_type == "Executor":
        return f"Executor: Executing '{task}' - Done."
    elif normalized_type == "Worker":
        return f"Worker: Handling routine task '{task}' - Processed with standard workflow."
    elif normalized_type == "Summarizer":
        return f"Summarizer: Concise summary of '{task}' - Key points extracted."
    elif normalized_type == "Verifier":
        return f"Verifier: Fact-checking '{task}' - Verified accurate."
    elif normalized_type == "Moderator":
        return f"Moderator: Guiding debate on '{task}' - Consensus reached."
    elif normalized_type == "Analyst":
        return f"Analyst: Reviewing logic for '{task}' - Analysis complete."
    elif normalized_type == "Coder":
        return f"Coder: Implementing '{task}' - Code generated."
    elif normalized_type == "Tester":
        return f"Tester: Testing '{task}' - Tests passed."
    elif normalized_type == "Security":
        return f"Security: Scanning '{task}' - Vulnerabilities addressed."
    elif normalized_type == "Documenter":
        return f"Documenter: Documenting '{task}' - Docs created."
    elif normalized_type == "Optimizer":
        return f"Optimizer: Optimizing '{task}' - Performance improved."
    else:
        return f"Default simulation for '{normalized_type}': Task '{task}' acknowledged."
def reflect_optimize(component: str, metrics: dict) -> str:
    """Simulate optimization based on metrics."""
    return f"Optimized {component} with metrics: {json.dumps(metrics)} - Adjustments applied."
def venv_create(env_name: str, with_pip: bool = True) -> str:
    """Create venv in sandbox."""
    safe_env = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, env_name)))
    if not safe_env.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Env path outside sandbox."
    try:
        venv.create(safe_env, with_pip=with_pip)
        return f"Venv '{env_name}' created."
    except Exception as e:
        return f"Venv error: {e}"
def restricted_exec(code: str, level: str = "basic") -> str:
    """Execute in restricted namespace."""
    restricted_ns = {"__builtins__": SAFE_BUILTINS.copy()} if level == "basic" else globals()
    try:
        exec(code, restricted_ns)
        return "Executed in restricted mode."
    except Exception as e:
        return f"Restricted exec error: {e}"
def isolated_subprocess(cmd: str, custom_env: dict = None) -> str:
    """Run in isolated subprocess."""
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, env=env, timeout=30)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Subprocess error: {e}"
def lisp_execution(code: str) -> str:
    """Execute Lisp code using Lispy interpreter."""
    try:
        # Lispy code from Norvig
        Symbol = str
        Number = (int, float)
        Atom = (Symbol, Number)
        List = list
        Exp = (Atom, List)
        Env = dict

        def tokenize(chars: str) -> list:
            return chars.replace('(', ' ( ').replace(')', ' ) ').split()

        def parse(program: str) -> Exp:
            return read_from_tokens(tokenize(program))

        def read_from_tokens(tokens: list) -> Exp:
            if len(tokens) == 0:
                raise SyntaxError('unexpected EOF')
            token = tokens.pop(0)
            if token == '(':
                L = []
                while tokens[0] != ')':
                    L.append(read_from_tokens(tokens))
                tokens.pop(0)
                return L
            elif token == ')':
                raise SyntaxError('unexpected )')
            else:
                return atom(token)

        def atom(token: str) -> Atom:
            try: return int(token)
            except ValueError:
                try: return float(token)
                except ValueError:
                    return Symbol(token)

        import math
        import operator as op

        def standard_env() -> Env:
            env = Env()
            env.update(vars(math))
            env.update({
                '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
                '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
                'abs':     abs,
                'append':  op.add,  
                'apply':   lambda proc, args: proc(*args),
                'begin':   lambda *x: x[-1],
                'car':     lambda x: x[0],
                'cdr':     lambda x: x[1:], 
                'cons':    lambda x,y: [x] + y,
                'eq?':     op.is_, 
                'expt':    pow,
                'equal?':  op.eq, 
                'length':  len, 
                'list':    lambda *x: List(x), 
                'list?':   lambda x: isinstance(x, List), 
                'map':     map,
                'max':     max,
                'min':     min,
                'not':     op.not_,
                'null?':   lambda x: x == [], 
                'number?': lambda x: isinstance(x, Number),  
                'print':   print,
                'procedure?': callable,
                'round':   round,
                'symbol?': lambda x: isinstance(x, Symbol),
            })
            return env

        class Env(dict):
            def __init__(self, parms=(), args=(), outer=None):
                self.update(zip(parms, args))
                self.outer = outer
            def find(self, var):
                return self if (var in self) else self.outer.find(var)

        class Procedure(object):
            def __init__(self, parms, body, env):
                self.parms, self.body, self.env = parms, body, env
            def __call__(self, *args): 
                return eval(self.body, Env(self.parms, args, self.env))

        global_env = standard_env()

        def eval(x, env=global_env):
            if isinstance(x, Symbol):
                return env.find(x)[x]
            elif not isinstance(x, List):
                return x   
            op, *args = x       
            if op == 'quote':
                return args[0]
            elif op == 'if':
                (test, conseq, alt) = args
                exp = (conseq if eval(test, env) else alt)
                return eval(exp, env)
            elif op == 'define':
                (symbol, exp) = args
                env[symbol] = eval(exp, env)
            elif op == 'set!':
                (symbol, exp) = args
                env.find(symbol)[symbol] = eval(exp, env)
            elif op == 'lambda':
                (parms, body) = args
                return Procedure(parms, body, env)
            else:
                proc = eval(op, env)
                vals = [eval(arg, env) for arg in args]
                return proc(*vals)

        return str(eval(parse(code)))
    except Exception as e:
        return f"Lisp execution error: {traceback.format_exc()}"
class GlyphEngine:
    def __init__(self):
        self.LOG_DIR = "glyph_data"
        os.makedirs(self.LOG_DIR, exist_ok=True)
        print("Loading semantic model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
        self.glyph_log = []
        self.id2glyph = {}
        self.glyph_graph = nx.Graph()
        self.graph_lock = threading.Lock()
        self.glyph_id = 0
        self.running = threading.Event()
        self.seen_signatures = set()
        self.reflex_free = []
        self.stagnant_counter = {}
        self.conceptual_attractors = {}
        self.temporal_clusters = {}
        self.dormant_glyphs = {}
        self.MAX_RAM_GLYPHS = 20_000
        self.LOW_ENTROPY = 100
        self.LOW_KEEP_PROB = 0.025
        self.PRUNE_AFTER = 250
        self.ATTRACTOR_THRESHOLD = 5
        self.DORMANCY_THRESHOLD = 500
        self.REACTIVATION_PROBABILITY = 0.1
        self.generation_count = 0
        self.current_season = GlyphEngine.SeasonalPhase.EXPLORATION
        self.season_duration = 1000
        self.season_counter = 0
        self.id_lock = threading.Lock()
        self.log_lock = threading.Lock()

    class ReflexType(Enum):
        DEFENSIVE = "defensive"
        EXPLORATORY = "exploratory"
        COLLABORATIVE = "collaborative"
        CONSOLIDATIVE = "consolidative"
        METAMORPHIC = "metamorphic"

    @dataclass
    class ReflexProfile:
        primary_type: 'GlyphEngine.ReflexType'
        intensity: float
        trigger_conditions: List[str]
        activation_count: int = 0
        last_activation: int = 0

    class SeasonalPhase(Enum):
        EXPLORATION = "exploration"
        CONSOLIDATION = "consolidation"
        DORMANCY = "dormancy"
        RENAISSANCE = "renaissance"

    def tag_vec(self, tag):
        return self.model.encode(tag)

    def embed(self, tags):
        if not tags:
            return np.zeros(384)
        vs = [self.tag_vec(t) for t in tags]
        return np.mean(vs, axis=0)

    def semantic_novelty(self, vec):
        if not self.glyph_log:
            return 1.0
        recent_sample = [g for g in self.glyph_log[-500:] if g is not None]
        if not recent_sample:
            return 1.0
        sample = random.sample(recent_sample, min(100, len(recent_sample)))
        dists = [cosine(vec, g['vec']) for g in sample if 'vec' in g]
        return max(dists) if dists else 1.0

    def productive_novelty(self, g):
        base_novelty = self.semantic_novelty(g['vec'])
        ancestry_bonus = 0
        for ancestor_id in g.get('ancestry', []):
            if ancestor_id in self.conceptual_attractors:
                ancestry_bonus += 0.1
        return base_novelty + ancestry_bonus

    def calculate_influence(self, g):
        if g['id'] not in self.glyph_graph:
            return 0
        children = []
        for neighbor in self.glyph_graph.neighbors(g['id']):
            neighbor_glyph = self.id2glyph.get(neighbor)
            if neighbor_glyph and g['id'] in neighbor_glyph.get('ancestry', []):
                children.append(neighbor_glyph)
        if not children:
            return 0
        offspring_novelty = sum(self.productive_novelty(child) for child in children) / len(children)
        tag_diversity = len(set().union(*[set(child['tags']) for child in children])) / max(1, len(children))
        max_depth = max((self.calculate_cascade_depth(child, visited=set()) for child in children), default=0)
        return offspring_novelty * 0.5 + tag_diversity * 0.3 + max_depth * 0.2

    def calculate_cascade_depth(self, g, visited=None, max_depth=7):
        if visited is None:
            visited = set()
        if g['id'] in visited or max_depth <= 0:
            return 0
        visited.add(g['id'])
        if g['id'] not in self.glyph_graph:
            return 1
        children = []
        for neighbor in self.glyph_graph.neighbors(g['id']):
            neighbor_glyph = self.id2glyph.get(neighbor)
            if neighbor_glyph and g['id'] in neighbor_glyph.get('ancestry', []):
                children.append(neighbor_glyph)
        if not children:
            return 1
        return 1 + max(self.calculate_cascade_depth(child, visited.copy(), max_depth-1) for child in children)

    def gen_id(self):
        with self.id_lock:
            self.glyph_id += 1
            return f"g{self.glyph_id:04}"

    def random_tag(self):
        base_tags = ['origin', 'flex', 'ghost', 'fractal', 'wild', 'mirror', 'unknown', 'stable']
        seasonal_tags = {
            self.SeasonalPhase.EXPLORATION: ['pioneer', 'venture', 'discover'],
            self.SeasonalPhase.CONSOLIDATION: ['anchor', 'strengthen', 'unify'],
            self.SeasonalPhase.DORMANCY: ['rest', 'potential', 'dormant'],
            self.SeasonalPhase.RENAISSANCE: ['reborn', 'transformed', 'awakened']
        }
        all_tags = base_tags + seasonal_tags.get(self.current_season, [])
        return random.choice(all_tags)

    def mutate_tag(self, tags):
        if len(tags) < 2:
            return 'm_misc'
        a, b = random.sample(tags, 2)
        if self.current_season == self.SeasonalPhase.EXPLORATION:
            return f"{a}→{b}"
        elif self.current_season == self.SeasonalPhase.CONSOLIDATION:
            return f"{a}∧{b}"
        else:
            return f"{a}+{b}"

    def calc_entropy(self, g, gen=None):
        if gen is None:
            gen = self.generation_count
        base = len(g['tags']) * 42 + random.randint(0, 58)
        gen_pressure = gen * 10
        seasonal_modifier = {
            self.SeasonalPhase.EXPLORATION: 1.2,
            self.SeasonalPhase.CONSOLIDATION: 0.8,
            self.SeasonalPhase.DORMANCY: 0.6,
            self.SeasonalPhase.RENAISSANCE: 1.5
        }
        return int(base + gen_pressure * seasonal_modifier.get(self.current_season, 1.0))

    def fitness(self, g):
        sem = self.productive_novelty(g)
        influence = self.calculate_influence(g)
        seasonal_bonus = 0
        if self.current_season == self.SeasonalPhase.EXPLORATION and sem > 0.7:
            seasonal_bonus = 50
        elif self.current_season == self.SeasonalPhase.CONSOLIDATION and g['id'] in self.conceptual_attractors:
            seasonal_bonus = 75
        elif self.current_season == self.SeasonalPhase.RENAISSANCE and 'renaissance' in g['tags']:
            seasonal_bonus = 100
        return 200 * sem - g['entropy'] + 50 * influence + seasonal_bonus

    def signature(self, g, bucket=25):
        return (tuple(sorted(g['tags'])), g['entropy'] // bucket)

    def is_novel(self, g):
        if g['entropy'] < self.LOW_ENTROPY:
            return random.random() < self.LOW_KEEP_PROB
        sig = self.signature(g)
        if sig in self.seen_signatures:
            return False
        self.seen_signatures.add(sig)
        return True

    def maybe_store(self, g):
        g['vec'] = self.embed(g['tags'])
        if self.is_novel(g):
            with self.log_lock:
                self.glyph_log.append(g)
                self.id2glyph[g['id']] = g
            with self.graph_lock:
                self.glyph_graph.add_node(g['id'], entropy=g['entropy'], tags=g['tags'], fit=self.fitness(g))
                self.stagnant_counter[g['id']] = 0
            if 'reflex' not in g['tags']:
                with self.log_lock:
                    self.reflex_free.append(g)

    def create_glyph(self):
        g = {
            'id': self.gen_id(),
            'tags': list({self.random_tag() for _ in range(random.randint(1, 3))}),
            'entropy': 0,
            'ancestry': [],
            'generation_born': self.generation_count
        }
        g['entropy'] = self.calc_entropy(g)
        g['vec'] = self.embed(g['tags'])
        return g

    def collide(self, a, b):
        child_tags = {*a['tags'], *b['tags'], self.mutate_tag(a['tags'] + b['tags'])}
        child = {
            'id': self.gen_id(),
            'tags': list(child_tags),
            'entropy': 0,
            'ancestry': [a['id'], b['id']],
            'generation_born': self.generation_count
        }
        child['entropy'] = self.calc_entropy(child)
        self.maybe_store(child)
        with self.graph_lock:
            self.glyph_graph.add_edge(a['id'], child['id'])
            self.glyph_graph.add_edge(b['id'], child['id'])
            self.stagnant_counter[a['id']] = 0
            self.stagnant_counter[b['id']] = 0
            self.stagnant_counter[child['id']] = 0
        self.temporal_clusters[child['id']] = self.generation_count

    def select_parents(self, k=10):
        with self.graph_lock:
            nodes = list(self.glyph_graph.nodes(data=True))
        if len(nodes) < 2:
            return []
        weights = []
        for node_id, node_data in nodes:
            base_weight = max(1, node_data.get('fit', 1))
            if node_id in self.conceptual_attractors:
                base_weight *= 2
            glyph = self.id2glyph.get(node_id)
            if glyph:
                if self.current_season == self.SeasonalPhase.EXPLORATION and self.semantic_novelty(glyph['vec']) > 0.5:
                    base_weight *= 1.5
                elif self.current_season == self.SeasonalPhase.CONSOLIDATION and len(glyph['tags']) > 2:
                    base_weight *= 1.3
            weights.append(base_weight)
        chosen = random.choices(nodes, weights=weights, k=min(k, len(nodes)))
        return [n[0] for n in chosen]

    def batch_collide(self, max_pairs=30):
        parents = self.select_parents(max_pairs)
        if len(parents) < 2:
            return
        for i in range(len(parents)):
            for j in range(i + 1, min(i + 5, len(parents))):
                a = self.id2glyph.get(parents[i])
                b = self.id2glyph.get(parents[j])
                if a and b:
                    self.collide(a, b)

    def determine_reflex_type(self, g, context_glyphs):
        if not context_glyphs:
            return self.ReflexType.EXPLORATORY
        entropy_ratio = g['entropy'] / max(1, np.mean([cg['entropy'] for cg in context_glyphs]))
        semantic_isolation = self.semantic_novelty(g['vec'])
        if entropy_ratio < 0.5 and semantic_isolation < 0.3:
            return self.ReflexType.DEFENSIVE
        elif semantic_isolation > 0.8:
            return self.ReflexType.EXPLORATORY
        elif len(g['tags']) > 3 and entropy_ratio > 1.2:
            return self.ReflexType.COLLABORATIVE
        elif g['id'] in self.conceptual_attractors:
            return self.ReflexType.CONSOLIDATIVE
        else:
            return self.ReflexType.METAMORPHIC

    def create_reflex_glyph(self, parent, reflex_type):
        base_tags = parent['tags'].copy()
        if reflex_type == self.ReflexType.DEFENSIVE:
            new_tags = base_tags + ['reflex', 'preserve', 'stable']
        elif reflex_type == self.ReflexType.EXPLORATORY:
            new_tags = base_tags + ['reflex', 'seek', 'novel', self.random_tag()]
        elif reflex_type == self.ReflexType.COLLABORATIVE:
            new_tags = base_tags + ['reflex', 'bridge']
        elif reflex_type == self.ReflexType.CONSOLIDATIVE:
            new_tags = base_tags + ['reflex', 'strengthen', 'anchor']
        else:
            new_tags = [self.mutate_tag(base_tags)] + ['reflex', 'transform', 'evolve']
        new_g = {
            'id': self.gen_id(),
            'tags': list(set(new_tags)),
            'entropy': 0,
            'ancestry': [parent['id']],
            'generation_born': self.generation_count
        }
        new_g['entropy'] = self.calc_entropy(new_g)
        return new_g

    def reflex_test(self):
        context_glyphs = self.glyph_log[-100:] if len(self.glyph_log) > 100 else self.glyph_log
        for g in [g for g in self.glyph_log if 'unknown' in g['tags'] and g['entropy'] < 150]:
            reflex_type = self.determine_reflex_type(g, context_glyphs)
            reflex_count = 2 if reflex_type == self.ReflexType.EXPLORATORY else 1
            for _ in range(reflex_count):
                new_g = self.create_reflex_glyph(g, reflex_type)
                self.maybe_store(new_g)
                with self.graph_lock:
                    self.glyph_graph.add_edge(g['id'], new_g['id'])

    def prune_stagnant(self):
        to_remove = []
        with self.graph_lock:
            for node in list(self.glyph_graph.nodes):
                self.stagnant_counter[node] = self.stagnant_counter.get(node, 0) + 1
                if self.stagnant_counter[node] > self.PRUNE_AFTER:
                    to_remove.append(node)
            for n in to_remove:
                self.glyph_graph.remove_node(n)
                self.stagnant_counter.pop(n, None)
        for nid in to_remove:
            self.id2glyph.pop(nid, None)
        ids_to_remove = set(to_remove)
        self.glyph_log[:] = [g for g in self.glyph_log if g['id'] not in ids_to_remove]

    def update_seasonal_phase(self):
        self.season_counter += 1
        if self.season_counter >= self.season_duration:
            self.season_counter = 0
            seasons = list(self.SeasonalPhase)
            current_index = seasons.index(self.current_season)
            self.current_season = seasons[(current_index + 1) % len(seasons)]
            print(f"\n🌍 Season change: {self.current_season.value} (Generation {self.generation_count})")

    def _jsonify(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple, set)):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._jsonify(v) for k, v in obj.items() if k not in ('vec', 'graph', 'nx_obj')}
        if isinstance(obj, (self.ReflexType, self.SeasonalPhase)):
            return obj.value
        if isinstance(obj, self.ReflexProfile):
            return {
                'primary_type': obj.primary_type.value,
                'intensity': obj.intensity,
                'trigger_conditions': obj.trigger_conditions,
                'activation_count': obj.activation_count
            }
        return str(obj)

    def save_logs(self, chunk_size=250):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        idx = len([f for f in os.listdir(self.LOG_DIR) if f.startswith('chunk_') and f.endswith('.json')])
        recent = self.glyph_log[-chunk_size:] if len(self.glyph_log) >= chunk_size else self.glyph_log
        with open(f"{self.LOG_DIR}/chunk_{idx:04}.json", "w", encoding="utf-8") as f:
            json.dump([self._jsonify(g) for g in recent], f, separators=(',', ':'))
        with open(f"{self.LOG_DIR}/master_log.json", "w", encoding="utf-8") as f:
            json.dump([self._jsonify(g) for g in self.glyph_log], f, separators=(',', ':'))
        print(f"💾 Saved to {self.LOG_DIR}/chunk_{idx:04}.json")

    def generate(self):
        while self.running.is_set():
            self.generation_count += 1
            if self.generation_count % 100 == 0:
                self.update_seasonal_phase()
            g = self.create_glyph()
            self.maybe_store(g)
            if self.generation_count % 5 == 0:
                self.batch_collide()
            if self.generation_count % 10 == 0:
                self.reflex_test()
            if self.generation_count % 50 == 0:
                self.prune_stagnant()
            if self.generation_count % 250 == 0 and len(self.glyph_log) > 0:
                self.save_logs()
                print(f"📊 Gen {self.generation_count}: {len(self.glyph_log)} glyphs, "
                      f"{self.glyph_graph.number_of_nodes()} nodes, "
                      f"{self.glyph_graph.number_of_edges()} edges")
            sleep_duration = {
                self.SeasonalPhase.EXPLORATION: 0.05,
                self.SeasonalPhase.CONSOLIDATION: 0.1,
                self.SeasonalPhase.DORMANCY: 0.2,
                self.SeasonalPhase.RENAISSANCE: 0.03
            }
            time.sleep(sleep_duration.get(self.current_season, 0.1))

    def start_system(self):
        self.running.set()
        print("🧠 Initializing Glyph Engine v4.0...")
        for _ in range(10):
            g = self.create_glyph()
            self.maybe_store(g)
        print(f"✅ Started with {len(self.glyph_log)} seed glyphs")
        print(f"🌍 Current season: {self.current_season.value}\n")
        self.generation_thread = threading.Thread(target=self.generate, daemon=True)
        self.generation_thread.start()

    def stop_system(self):
        self.running.clear()
        self.save_logs()
        print("\n🛑 Glyph Engine stopped")

    def get_status(self):
        return {
            "generation_count": self.generation_count,
            "glyph_count": len(self.glyph_log),
            "nodes": self.glyph_graph.number_of_nodes(),
            "edges": self.glyph_graph.number_of_edges(),
            "season": self.current_season.value,
            "running": self.running.is_set()
        }

    def get_recent_glyphs(self, n=5):
        return [self._jsonify(g) for g in self.glyph_log[-n:]]


glyph_engine = None

def glyph_control(action: str, params: dict = None) -> str:
    global glyph_engine
    if glyph_engine is None:
        glyph_engine = GlyphEngine()
    if action == "start":
        glyph_engine.start_system()
        return "Glyph Engine started."
    elif action == "stop":
        glyph_engine.stop_system()
        return "Glyph Engine stopped."
    elif action == "status":
        return json.dumps(glyph_engine.get_status())
    elif action == "get_recent_glyphs":
        n = params.get("n", 5) if params else 5
        return json.dumps(glyph_engine.get_recent_glyphs(n))
    else:
        return "Unknown action."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read file content from sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to file."}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_file",
            "description": "Write content to file in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list_files",
            "description": "List files in directory within sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Relative dir path (default root)."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_mkdir",
            "description": "Create directory in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Relative dir path."}
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch current time (sync optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sync": {"type": "boolean", "description": "True for NTP sync (default False)."},
                    "format": {"type": "string", "description": "Format: iso (default), human, json."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute Python code in REPL (venv optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "venv_path": {"type": "string", "description": "Optional venv path."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": "Insert/update memory entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Memory key."},
                    "mem_value": {"type": "object", "description": "Value dict."},
                },
                "required": ["mem_key", "mem_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_query",
            "description": "Query memory by key or recent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Specific key to query (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max recent entries if no key (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_ops",
            "description": "Basic Git operations in sandbox (init, commit, branch, diff). No remote operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["init", "commit", "branch", "diff"]},
                    "repo_path": {"type": "string", "description": "Relative path to repo."},
                    "message": {
                        "type": "string",
                        "description": "Commit message (for commit).",
                    },
                    "name": {"type": "string", "description": "Branch name (for branch)."},
                },
                "required": ["operation", "repo_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_query",
            "description": "Interact with local SQLite database in sandbox (create, insert, query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {"type": "string", "description": "Relative path to DB file."},
                    "query": {"type": "string", "description": "SQL query."},
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Query parameters.",
                    },
                },
                "required": ["db_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run safe whitelisted shell commands in sandbox (e.g., ls, grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command string."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_lint",
            "description": "Lint and auto-format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language (python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust).",
                    },
                    "code": {"type": "string", "description": "Code snippet."},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_simulate",
            "description": "Simulate API calls with mock or fetch from public APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API URL."},
                    "method": {"type": "string", "description": "GET/POST (default GET)."},
                    "data": {"type": "object", "description": "POST data."},
                    "mock": {"type": "boolean", "description": "True for mock (default)."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_consolidate",
            "description": "Brain-like consolidation: Summarize and embed data for hierarchical storage. Use for chat logs to create semantic summaries and episodic details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Key for the memory entry."},
                    "interaction_data": {
                        "type": "object",
                        "description": "Data to consolidate (dict).",
                    },
                },
                "required": ["mem_key", "interaction_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_retrieve",
            "description": "Retrieve relevant memories via embedding similarity. Use before queries to augment context efficiently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query string for similarity search."},
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_prune",
            "description": "Prune low-salience memories to optimize storage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "langsearch_web_search",
            "description": "Search the web using LangSearch API for relevant results, snippets, and optional summaries. Supports time filters and limits up to 10 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (supports operators like site:example.com).",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear, or noLimit (default).",
                        "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"],
                    },
                    "summary": {"type": "boolean", "description": "Include long text summaries (default True)."},
                    "count": {"type": "integer", "description": "Number of results (1-10, default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_embedding",
            "description": "Generate vector embedding for text using SentenceTransformer (384-dim vector). Use for semantic processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Perform ANN vector search in ChromaDB using cosine similarity. Returns top matches above threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query embedding vector.",
                    },
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (default 0.6).",
                    },
                },
                "required": ["query_embedding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_text",
            "description": "Split text into semantic chunks (default 512 tokens) for processing large inputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to chunk."},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens per chunk (default 512).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_chunk",
            "description": "Compress a text chunk via LLM summary (under 100 words), preserving key facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk": {"type": "string", "description": "Text chunk to summarize."}
                },
                "required": ["chunk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Keyword-based search on memory cache (simple overlap/BM25 sim). Returns top matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "socratic_api_council",
            "description": "Run a BTIL/MAD-enhanced Socratic Council with multiple personas (Planner, Critic, Executor, Summarizer, Verifier, Moderator) via xAI API for debate and consensus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of branch options to evaluate.",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model (default: grok-4-fast-reasoning).",
                    },
                    "user": {
                        "type": "string",
                        "description": "User for memory consolidation (required).",
                    },
                    "convo_id": {
                        "type": "integer",
                        "description": "Conversation ID for memory (default: 0).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (optional, uses global if not provided).",
                    },
                },
                "required": ["branches", "user"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_spawn",
            "description": "Spawn sub-agent (Planner, Critic, Executor, Worker, Summarizer, Verifier, Moderator, Analyst, Coder, Tester, Security, Documenter, Optimizer, or custom) for task. Case-insensitive, custom types allowed with fallback simulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_type": {
                        "type": "string",
                        "description": "Type: Planner, Critic, Executor, Worker, Summarizer, Verifier, Moderator, Analyst, Coder, Tester, Security, Documenter, Optimizer (case-insensitive) or custom."
                    },
                    "task": {
                        "type": "string",
                        "description": "Task for sub-agent."
                    }
                },
                "required": ["sub_agent_type", "task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reflect_optimize",
            "description": "Optimize component based on metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Component to optimize (e.g., prompt)."
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Performance metrics dict."
                    }
                },
                "required": ["component", "metrics"]
            }
        }
    },
    # New tool schemas
    {
        "type": "function",
        "function": {
            "name": "venv_create",
            "description": "Create a virtual Python environment in sandbox for isolated package installations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Name of the venv."},
                    "with_pip": {"type": "boolean", "description": "Include pip (default True)."}
                },
                "required": ["env_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restricted_exec",
            "description": "Execute code in a restricted namespace with optional levels (basic/full).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "level": {"type": "string", "description": "Access level: basic (default) or full."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolated_subprocess",
            "description": "Run command in an isolated subprocess with custom env.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run."},
                    "custom_env": {"type": "object", "description": "Custom environment variables."}
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lisp_execution",
            "description": "Execute Lisp code using a built-in Lisp interpreter (Lispy). Supports basic Scheme-like expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Lisp code to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glyph_control",
            "description": "Control the Glyph Engine: start, stop, status, get_recent_glyphs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop", "status", "get_recent_glyphs"]},
                    "params": {"type": "object", "description": "Additional params, e.g., {'n': 5} for get_recent_glyphs."},
                },
                "required": ["action"],
            },
        },
    },
]
# Tool Dispatcher Dictionary
TOOL_DISPATCHER = {
    "fs_read_file": fs_read_file,
    "fs_write_file": fs_write_file,
    "fs_list_files": fs_list_files,
    "fs_mkdir": fs_mkdir,
    "get_current_time": get_current_time,
    "code_execution": code_execution,
    "memory_insert": memory_insert,
    "memory_query": memory_query,
    "git_ops": git_ops,
    "db_query": db_query,
    "shell_exec": shell_exec,
    "advanced_memory_consolidate": advanced_memory_consolidate,
    "advanced_memory_retrieve": advanced_memory_retrieve,
    "advanced_memory_prune": advanced_memory_prune,
    "code_lint": code_lint,
    "api_simulate": api_simulate,
    "langsearch_web_search": langsearch_web_search,
    "generate_embedding": generate_embedding,
    "vector_search": vector_search,
    "chunk_text": chunk_text,
    "summarize_chunk": summarize_chunk,
    "keyword_search": keyword_search,
    "socratic_api_council": socratic_api_council,
    "agent_spawn": agent_spawn,
    "reflect_optimize": reflect_optimize,
    "venv_create": venv_create,
    "restricted_exec": restricted_exec,
    "isolated_subprocess": isolated_subprocess,
    "lisp_execution": lisp_execution,
    "glyph_control": glyph_control,
}
tool_count = 0
council_count = 0
main_count = 0
def call_xai_api(
    model,
    messages,
    sys_prompt,
    stream=True,
    image_files=None,
    enable_tools=False,
):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
    api_messages = [{"role": "system", "content": sys_prompt}]
    # Add history
    for msg in messages:
        content_parts = [{"type": "text", "text": msg["content"]}]
        # Add images only to the last user message
        if msg["role"] == "user" and image_files and msg is messages[-1]:
            for img_file in image_files:
                img_file.seek(0)
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_file.type};base64,{img_data}"},
                    }
                )
        api_messages.append(
            {
                "role": msg["role"],
                "content": content_parts if len(content_parts) > 1 else msg["content"],
            }
        )
    def generate(current_messages):
        global tool_count, council_count, main_count
        max_iterations = 10
        for _ in range(max_iterations):
            main_count += 1
            print(f"\rTools: {tool_count} | Council: {council_count} | Main: {main_count}", end='')
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=TOOLS if enable_tools else None,
                    tool_choice="auto" if enable_tools else None,
                    stream=True,
                )
                tool_calls = []
                full_delta_response = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        full_delta_response += delta.content
                    if delta and delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)
                if not tool_calls:
                    break # Exit loop if no tools are called
                current_messages.append(
                    {"role": "assistant", "content": full_delta_response, "tool_calls": tool_calls}
                )
                yield "\n*Thinking... Using tools...*\n"
                tool_outputs = []
                conn.execute("BEGIN")
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                        func_to_call = TOOL_DISPATCHER.get(func_name)
                        if (
                            func_name.startswith("memory")
                            or func_name.startswith("advanced_memory")
                            or func_name == "socratic_api_council"
                            or func_name == "agent_spawn"
                            or func_name == "reflect_optimize"
                        ):
                            args["user"] = st.session_state["user"]
                            args["convo_id"] = st.session_state.get("current_convo_id", 0)
                        # Override model for socratic_api_council with UI selection
                        if func_name == "socratic_api_council":
                            args["model"] = st.session_state.get(
                                "council_model_select", "grok-4-fast-reasoning"
                            )
                        if func_to_call:
                            result = func_to_call(**args)
                        else:
                            result = f"Unknown tool: {func_name}"
                    except Exception as e:
                        result = f"Error calling tool {func_name}: {e}"
                    if func_name == "socratic_api_council":
                        council_count += 1
                    else:
                        tool_count += 1
                    print(f"\rTools: {tool_count} | Council: {council_count} | Main: {main_count}", end='')
                    if func_name == "socratic_api_council":
                        print(f"\nCouncil Call {council_count}: {result}")
                    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "role": "tool", "content": str(result)}
                    )
                conn.commit()
                current_messages.extend(tool_outputs)
            except Exception as e:
                error_msg = f"API or Tool Error: {traceback.format_exc()}"
                yield f"\nAn error occurred: {e}. Aborting this turn."
                st.error(error_msg)
                break
    return generate(api_messages)
# Login Page
def login_page():
    st.title("Welcome to The Apex Orchestrator Interface")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                c.execute("SELECT password FROM users WHERE username=?", (username,))
                result = c.fetchone()
                if result and verify_password(result[0], password):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Register"):
                c.execute("SELECT * FROM users WHERE username=?", (new_user,))
                if c.fetchone():
                    st.error("Username already exists.")
                else:
                    c.execute(
                        "INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass))
                    )
                    conn.commit()
                    st.success("Registered! Please login.")
def load_history(convo_id):
    """Loads a specific conversation from the database into the session state."""
    c.execute(
        "SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state["user"])
    )
    result = c.fetchone()
    if result:
        messages = json.loads(result[0])
        st.session_state["messages"] = messages
        st.session_state["current_convo_id"] = convo_id
        st.rerun()
def delete_history(convo_id):
    """Deletes a specific conversation from the database."""
    c.execute(
        "DELETE FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state["user"])
    )
    conn.commit()
    # If the deleted chat was the one currently loaded, clear the session
    if st.session_state.get("current_convo_id") == convo_id:
        st.session_state["messages"] = []
        st.session_state["current_convo_id"] = 0
    st.rerun()
def chat_page():
    st.title(f"Apex Chat - {st.session_state['user']}")
    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Chat Settings")
        model = st.selectbox(
            "Select Model",
            ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"],
            key="model_select",
        )
        council_model = st.selectbox(
            "Select Council Model",
            ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"],
            key="council_model_select",
        )
        prompt_files = load_prompt_files()
        if prompt_files:
            selected_file = st.selectbox(
                "Select System Prompt", prompt_files, key="prompt_select"
            )
            with open(os.path.join(PROMPTS_DIR, selected_file), "r") as f:
                prompt_content = f.read()
            custom_prompt = st.text_area(
                "Edit System Prompt", value=prompt_content, height=200, key="custom_prompt"
            )
        else:
            st.warning("No prompt files found in ./prompts/")
            custom_prompt = st.text_area(
                "System Prompt", value="You are a helpful AI.", height=200, key="custom_prompt"
            )
        # The key for the file uploader is important
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=["jpg", "png"],
            accept_multiple_files=True,
            key="uploaded_images",
        )
        enable_tools = st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
        st.divider()
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["current_convo_id"] = 0
            st.rerun()
        st.header("Chat History")
        c.execute(
            "SELECT convo_id, title FROM history WHERE user=? ORDER BY convo_id DESC",
            (st.session_state["user"],),
        )
        histories = c.fetchall()
        for convo_id, title in histories:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(title, key=f"load_{convo_id}", use_container_width=True):
                    load_history(convo_id)
            with col2:
                if st.button("🗑️", key=f"delete_{convo_id}", use_container_width=True):
                    delete_history(convo_id)
    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
    if prompt := st.chat_input("Your command, ape?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        with st.chat_message("assistant"):
            images_to_process = uploaded_images if uploaded_images else []
            generator = call_xai_api(
                model,
                st.session_state.messages,
                custom_prompt,
                stream=True,
                image_files=images_to_process,
                enable_tools=enable_tools,
            )
            full_response = st.write_stream(generator)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Save to History
        title_message = next(
            (msg["content"] for msg in st.session_state.messages if msg["role"] == "user"),
            "New Chat",
        )
        title = (title_message[:40] + "...") if len(title_message) > 40 else title_message
        messages_json = json.dumps(st.session_state.messages)
        if st.session_state.get("current_convo_id", 0) == 0:
            c.execute(
                "INSERT INTO history (user, title, messages) VALUES (?, ?, ?)",
                (st.session_state["user"], title, messages_json),
            )
            st.session_state.current_convo_id = c.lastrowid
        else:
            c.execute(
                "UPDATE history SET title=?, messages=? WHERE convo_id=?",
                (title, messages_json, st.session_state.current_convo_id),
            )
        conn.commit()
        st.rerun()
# Main App Logic
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state.get("logged_in"):
        chat_page()
    else:
        login_page()
