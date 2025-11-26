import os
from config import PDF_PATH, HF_TOKEN, HF_MODEL, TOP_K
from pdf_handler import extract_text_from_pdf, chunk_text
from embeddings import init_embedding_model, embed_texts
from pinecone_client import upsert_chunks, query_pinecone
import numpy as np
from huggingface_hub import InferenceClient

def ingest_pdf(pc_index):
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    texts = [c[0] for c in chunks] #Extracts only the text part from each chunk tuple

    model = init_embedding_model()

    print(f"Embedding {len(texts)} chunks...")
    embeddings = []
    BATCH_SIZE = 64
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_emb = embed_texts(model, batch)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings) #Stacks all batch embeddings into a single 2D array.

    filename = os.path.basename(PDF_PATH).replace(" ", "_")
    ids = [f"{filename}_chunk_{i}" for i in range(len(texts))]
    metadatas = [
        {"source": filename, "start": chunks[i][1], "end": chunks[i][2], "text_preview": texts[i][:200]}
        for i in range(len(texts))
    ]

    print("Upserting into Pinecone...")
    upsert_chunks(pc_index, ids, embeddings, metadatas)
    print(f"✓ Ingested {len(ids)} chunks from {PDF_PATH}")

def answer_query(pc_index, query):
    model = init_embedding_model()
    q_emb = embed_texts(model, [query])[0]

    matches = query_pinecone(pc_index, q_emb, top_k=TOP_K)

    context_blocks = []
    for m in matches:
        md = m.metadata or {}
        context_blocks.append(f"SOURCE: {md.get('source','unknown')}\n{md.get('text_preview','')}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer:

CONTEXT:
{context}

QUESTION: {query}

If you cannot find the answer in the context, say "Not enough information."
"""

    client = InferenceClient(token=HF_TOKEN)
    answer = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = answer.choices[0].message["content"].strip()
    return answer, matches
