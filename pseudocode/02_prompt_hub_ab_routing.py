"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path

# ── 1. Environment / imports ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGSMITH_PROJECT", "day22-lab")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
# TODO: write PROMPT_V1 — concise, 2-4 sentence answers
# SYSTEM_V1 = (
#     "You are a helpful AI assistant. "
#     "Answer the user's question using ONLY the provided context. "
#     "Keep your answer concise (2-4 sentences). "
#     "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
#     "Context:\n{context}"
# )
# PROMPT_V1 = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_V1),
#     ("human",  "{question}"),
# ])

# TODO: write PROMPT_V2 — structured, expert 3-5 sentence answers
# SYSTEM_V2 = (
#     "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
#     "Instructions:\n"
#     "1. Read the context carefully.\n"
#     "2. Identify the key facts relevant to the question.\n"
#     "3. Write a clear, well-organized answer (3-5 sentences).\n"
#     "4. State explicitly if the context lacks sufficient information.\n\n"
#     "Context:\n{context}"
# )
# PROMPT_V2 = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_V2),
#     ("human",  "{question}"),
# ])

# Prompt Hub names (change these to your own unique names)
PROMPT_V1_NAME = "my-rag-prompt-v1"   # TODO: choose a unique name
PROMPT_V2_NAME = "my-rag-prompt-v2"   # TODO: choose a unique name

SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.

    Use: client.push_prompt(name, object=template, description="...")
    The 'object' argument must be a ChatPromptTemplate instance.
    """
    # Push PROMPT_V1
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 - concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1: {e}")

    # Push PROMPT_V2
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 - structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.

    Use: client.pull_prompt(name) → returns a ChatPromptTemplate
    """
    prompts = {}

    # Pull PROMPT_V1_NAME, fall back to local PROMPT_V1 on error
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    # Pull PROMPT_V2_NAME, fall back to local PROMPT_V2 on error
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.

    Rules:
      even hash → PROMPT_V1_NAME
      odd  hash → PROMPT_V2_NAME

    This is DETERMINISTIC: same request_id always maps to the same version.
    """
    # Compute MD5 hash of request_id, convert to integer
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # Return V1 name if even, V2 name if odd
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    """Build vectorstore using same logic as step 1"""
    dataset_path = Path("data/knowledge_base.txt")
    if not dataset_path.exists():
        text = """Machine Learning Basics. LLMs in 2024. RAG systems. Vector databases."""
    else:
        text = dataset_path.read_text()
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"), # type: ignore
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}

    Steps:
      a) Retrieve top-3 docs with retriever.invoke(question)
      b) Join their page_content into a single context string
      c) Run (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Return the result dict
    """
    # Retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # Return result
    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # Create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # Push both prompts
    push_prompts_to_hub(client)

    # Pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"), #type: ignore
    )

    # Import sample questions from step 1
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).parent))
    step1_module = import_module("01_langsmith_rag_pipeline")
    SAMPLE_QUESTIONS = step1_module.SAMPLE_QUESTIONS
    
    v1_count, v2_count = 0, 0
    print("\nRouting 50 questions through A/B prompts...")
    
    # Loop over all 50 questions with A/B routing
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}/50] [prompt-{version_tag}] {question[:55]}...")

    print(f"\n✅ Routing Summary:")
    print(f"   V1 (concise): {v1_count} questions")
    print(f"   V2 (structured): {v2_count} questions")


if __name__ == "__main__":
    main()
