from embeddings import init_embedding_model
from pinecone_client import init_pinecone, create_index_if_missing
from rag_pipeline import ingest_pdf, answer_query
from config import PINECONE_INDEX, PDF_PATH, QUERY

def main():
    # Initialize Pinecone
    pc = init_pinecone()

    # Determine embedding dimension
    emb_model = init_embedding_model()
    dim = emb_model.get_sentence_embedding_dimension()

    # Create index if missing
    index = create_index_if_missing(pc, PINECONE_INDEX, dim)

    # Ingest PDF
    ingest_pdf(index)

    # Run RAG query
    answer, retrieved = answer_query(index, QUERY)

    # print("\n--- Retrieved Chunks ---")
    # for m in retrieved:
    #     print(m)

    print("\n--- LLM Answer ---\n")
    print(answer)

if __name__ == "__main__":
    main()
