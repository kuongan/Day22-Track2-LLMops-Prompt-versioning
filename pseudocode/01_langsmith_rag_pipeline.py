"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
from pathlib import Path

# ── 1. Environment setup ────────────────────────────────────────────────────
# Load your .env file using python-dotenv
from dotenv import load_dotenv
load_dotenv()

# Set LangSmith environment variables BEFORE importing LangChain
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGSMITH_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
# Create a ChatOpenAI instance
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"), # type: ignore
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# Create an OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"), # type: ignore
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Steps:
      a) Read your dataset
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Call FAISS.from_texts(chunks, embeddings) to build the index
      d) Return the vectorstore
    """
    # Read your dataset file
    dataset_path = Path("data/knowledge_base.txt")
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}. Using sample data.")
        text = """Machine Learning is a subset of Artificial Intelligence. 
                  The three main types are supervised learning, unsupervised learning, and reinforcement learning.
                  Deep learning uses neural networks. Transfer learning reuses pre-trained models."""
    else:
        text = dataset_path.read_text()

    # Create a text splitter and split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"✓ Split into {len(chunks)} chunks")

    # Build and return the FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
# Define a ChatPromptTemplate with:
#   - system message: instruct the LLM to answer using ONLY the provided context
#   - human message: the user's {question}
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    # Create a retriever from the vectorstore (k=3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define a helper to join retrieved docs into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build and return the LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
# Decorate this function with @traceable so LangSmith captures it
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


# ── 8. Sample questions (50 total — one per topic area) ────────────────────
SAMPLE_QUESTIONS = [
    "What are the three main types of machine learning?",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "How does regularization prevent overfitting?",
    "What is cross-validation?",
    "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?",
    "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?",
    "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?",
    "What are word embeddings?",
    "What is transfer learning in NLP?",
    "How does BERT handle language understanding?",
    "What is self-attention in transformers?",
    "What is GPT and how is it trained?",
    "What is instruction tuning?",
    "What is RLHF?",
    "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?",
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
    "What is dense retrieval?",
    "Why is chunking strategy important in RAG?",
    "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?",
    "What is FAISS?",
    "How do text embeddings capture semantic meaning?",
    "What is HNSW?",
    "What is hybrid search in vector databases?",
    "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?",
    "What is LangGraph?",
    "What memory types does LangChain support?",
    "What are LangChain retrievers?",
    "What is LangSmith?",
    "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?",
    "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?",
    "What is RAGAS?",
    "How does RAGAS compute faithfulness?",
    "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?",
    "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?",
    "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?",
    "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # Build the vectorstore
    vectorstore = build_vectorstore()

    # Build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # Loop through all SAMPLE_QUESTIONS, call ask(), print results
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    # Print confirmation that traces were sent
    print(f"{len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
