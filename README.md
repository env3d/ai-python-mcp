# Simple RAG (Retrieval Augmented Generation)

This repository demonstrates a very small Retrieval-Augmented Generation (RAG) example:
- A retriever that builds and queries a FAISS index of text embeddings.
- A generator (chat) that injects retrieved passages into a prompt.

## Files and key symbols
- Retriever: [rag.py](rag.py) — provides the simple search function [`rag.search`](rag.py) and index bootstrap [`rag.load_index`](rag.py). It uses `sentence-transformers` and FAISS to encode and index the corpus.
- Corpus: [kjv.txt](kjv.txt) — the plain-text source used to build the FAISS index.
- Chat/generation: [chat.py](chat.py) — exposes functions such as [`chat.complete`](chat.py) and chat utilities that the app calls.
- App entrypoint: [main.py](main.py) — a minimal chatbot that injects the retrieved passages into the template variable [`main.chat_template`](main.py) and drives the loop via [`main.main`](main.py).
- Index file (generated): [faiss_index.bin](faiss_index.bin) — the serialized FAISS index created by [`rag.load_index`](rag.py).

## How it works (quick)
1. `rag.load_index()` will load [faiss_index.bin](faiss_index.bin) if present. If not, it reads [kjv.txt](kjv.txt), encodes the lines with the sentence-transformers model, creates a FAISS index, and writes [faiss_index.bin](faiss_index.bin).
2. `rag.search(query, top_k)` returns the top-k matching lines from the corpus.
3. [main.py](main.py) calls [`rag.search`](rag.py) to collect context and passes that context into the chat prompt/template, then calls [`chat.complete`](chat.py) to generate a response.

## Assignment
- Create your own specialized chatbot
- Replace the text file [kjv.txt](kjv.txt) with your own corpus (one document or one line per passage).
- Remove the existing index to force regeneration
- Make appropriate changes to the chat template

## Submit

Make sure you are satisified with your work by running `pytest`.

When you are ready to submit, execute the following commands in the terminal:

```bash
$ git add -A
$ git commit -m 'submit'
$ git push
```