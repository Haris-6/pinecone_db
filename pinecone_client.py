from pinecone import Pinecone, ServerlessSpec
"""
pinecone Connect to Pinecone vector database.
ServerlessSpec: Define cloud region and serverless settings for the index.
"""
from config import PINECONE_API_KEY, PINECONE_INDEX, BATCH_SIZE

#batch size for upserts is use to control how many vectors are sent to Pinecone in a single request.

def init_pinecone():
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY")
    return Pinecone(api_key=PINECONE_API_KEY) #Creates a Pinecone client instance. 

def create_index_if_missing(pc, index_name, dim):
    # dim: Dimension of the vectors (embedding size).
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") #Creates a serverless index on AWS, meaning Pinecone handles scaling automatically.
        )
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)  #Returns a handle to the index for upserting and querying vectors.

def upsert_chunks(pc_index, ids, vectors, metadatas, batch_size=BATCH_SIZE):
    for i in range(0, len(ids), batch_size): #: Loops through vectors in chunks of size batch_size.
        j = min(i + batch_size, len(ids)) #min(i + batch_size, len(ids)): Ensures the last batch does not go out of bounds.
        batch = [
            {"id": ids[k], "values": vectors[k].tolist(), "metadata": metadatas[k]}
            for k in range(i, j)
        ]
        pc_index.upsert(vectors=batch) #Uploads the batch to Pinecone.

def query_pinecone(pc_index, query_vector, top_k=5):
    result = pc_index.query(
        vector=query_vector.tolist(), #Query vector as a list.
        top_k=top_k,
        include_metadata=True
    )
    return result.matches
