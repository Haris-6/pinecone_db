import os
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()
# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(api_key=api_key)
client1=InferenceClient(token=HF_TOKEN)
# -------------------------------
# STEP 1 — Create a vector store
# -------------------------------
# store_name = "my_api_headers_store"
# store = client.vector_stores.create(name=store_name)
# print(f"Vector store created with ID: {store.id}")

# # -------------------------------
# # STEP 2 — Upload JSON data
# # -------------------------------
# # Make sure you have a JSON file, e.g., headers.json
# file_streams = [open("./headers.json", "rb")]
# file_batch = client.vector_stores.file_batches.upload_and_poll(
#     vector_store_id=store.id,
#     files=file_streams
# )
# print("Upload status:", file_batch.status)

# -------------------------------
# STEP 3 — Query the vector store
# -------------------------------
# storeid="vs_6926ba43ecc88191a3643320402fe246"
# 3. Search in the vector store
query = input("please enter the query : ")
search_res = client.vector_stores.search(
    vector_store_id=storeid,
    query=query,
    max_num_results=3
)

#print("Search results:", search_res.data)   # list of matching chunks

# 4. Use results as context and ask the model
if search_res.data:
    # Combine content from top result(s)
    # Use attribute access
    context = "\n".join([item.content[0].text for item in search_res.data])


    resp = client1.chat.completions.create(
        model="qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "Use the following context to answer:"},
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    )
    print("Answer:", resp.choices[0].message.content)
else:
    print("No relevant context found.")