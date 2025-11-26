import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -------------------------------
# STEP 1 — Create a vector store
# -------------------------------
store_name = "my_api_headers_store"
store = client.vector_stores.create(name=store_name)
print(f"Vector store created with ID: {store.id}")

# -------------------------------
# STEP 2 — Upload JSON data
# -------------------------------
# Make sure you have a JSON file, e.g., headers.json
file_streams = [open("./headers.json", "rb")]
file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=store.id,
    files=file_streams
)
print("Upload status:", file_batch.status)

# -------------------------------
# STEP 3 — Query the vector store
# -------------------------------

# 3. Search in the vector store
query = "Find the API that uses Bearer tokens"
search_res = client.vector_stores.search(
    vector_store_id=store.id,
    query=query,
    max_num_results=3
)

print("Search results:", search_res.data)   # list of matching chunks

# 4. Use results as context and ask the model
if search_res.data:
    # Combine content from top result(s)
    # Use attribute access
    context = "\n".join([item.content[0].text for item in search_res.data])


    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Use the following context to answer:"},
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    )
    print("Answer:", resp.choices[0].message.content)
else:
    print("No relevant context found.")