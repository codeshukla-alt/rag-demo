# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique for grounding a large language model (LLM) in
external knowledge. Instead of relying only on what the model memorized during training, a RAG system
first retrieves relevant documents from a knowledge base and then passes them to the model as context
when generating an answer.

## Why use RAG
RAG addresses three common LLM problems:
1. Hallucination: by giving the model source text to ground its answer, RAG reduces fabricated facts.
2. Staleness: the knowledge base can be updated without retraining the model.
3. Attribution: because answers are drawn from retrieved passages, they can cite their sources.

## Core pipeline
A typical RAG pipeline has these stages:
- Ingestion: load source documents (PDFs, web pages, markdown) and split them into chunks.
- Embedding: convert each chunk into a numeric vector using an embedding model.
- Indexing: store the vectors in a vector database such as FAISS, pgvector, or Pinecone.
- Retrieval: embed the user's question and find the most similar chunks (top-k) via cosine similarity.
- Generation: build a prompt containing the retrieved chunks and ask the LLM to answer using only that
  context, citing the chunks it used.

## Chunking
Chunking splits long documents into smaller passages so retrieval is precise. A common approach is
fixed-size chunks (for example 800 characters) with an overlap (for example 120 characters) so that
sentences spanning a boundary are not lost. Smaller chunks improve precision; larger chunks preserve
more context.

## Improving quality
Retrieval quality can be improved with hybrid search (combining keyword BM25 with vector search),
re-ranking retrieved results with a cross-encoder, and query rewriting. Answer quality should always
be measured with an evaluation set rather than judged by eye.
