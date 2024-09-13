from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.redis import RedisVectorStore
from dotenv import load_dotenv
import os
import redis
import json
from redisvl.schema import IndexSchema
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import numpy as np
import time
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

def create_redis_schema(name, prefix):
    return IndexSchema.from_dict({
        'index': {
            'name': name,
            'prefix': prefix,
            'key_separator': ':'
        },
        'fields': [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {"type": "numeric", "name": "updated_at"},
            {"type": "tag", "name": "file_name"},
            {
                'type': 'vector',
                'name': 'vector',
                'attrs': {
                    'dims': 384,
                    'algorithm': 'hnsw',
                    'distance_metric': 'cosine',
                },
            }
        ]
    })


def get_cached_response(cache_vector_store, query_embedding, similarity_threshold=0.9):
    """
    Retrieve a cached response for a similar query based on query embedding from the cache.

    Parameters:
    - cache_vector_store: The vector store containing cached embeddings and responses.
    - query_embedding: The embedding of the current query to find in the cache.
    - similarity_threshold: The cutoff similarity score to consider two embeddings as similar.

    Returns:
    - The cached response if a similar query is found, otherwise None.
    """
    # Search for similar embeddings in the cache vector store
    results = cache_vector_store.similar(query_embedding, similarity_cutoff=similarity_threshold, k=1)
    
    # If a similar query is found, return the first result
    if results:
        print("Cached response found!")
        return results[0]  # Return the cached document

    # If no similar query is found, return None
    print("No cached response found for this query.")
    return None


def main():
    load_dotenv()

    redis_client = SET_YOUR_OWN

    try:
        redis_client.ping()
        print("Connected to Redis successfully!")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    documents = SimpleDirectoryReader(input_dir='./data').load_data(num_workers=4)
    print(f"Loaded {len(documents)} documents")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(api_key=os.getenv("GEMINI_API_KEY"), model="models/gemini-pro")

    # Create schemas for both document index and cache
    doc_schema = create_redis_schema("financial_data", "data")
    cache_schema = create_redis_schema("Cache", "cache")

    # Initialize vector stores for both documents and cache
    doc_vector_store = RedisVectorStore(schema=doc_schema, redis_client=redis_client, overwrite=True)
    cache_vector_store = RedisVectorStore(schema=cache_schema, redis_client=redis_client, overwrite=True)

    storage_context = StorageContext.from_defaults(vector_store=doc_vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

    query_engine = index.as_query_engine()

    while True:
        user_query = input("Enter your query (type 'exit' to end): ")
        if user_query.lower() == "exit":
            print("Ending the query engine. Goodbye!")
            break

        query_embedding = Settings.embed_model.get_text_embedding(user_query)

        # Check for similar query in cache
        
        similar_query = get_cached_response(cache_vector_store, query_embedding)

        if similar_query:
            print("Response from Cache:")
            print(similar_query)
        else:
            # Execute the query and print the response
            response = query_engine.query(user_query)
            print("Response from Query Engine:")
            print(response)

            # Cache the response
            cache_vector_store.add(
                np.array([query_embedding]),                
                documents=[{
                    "id": str(hash(user_query)),
                    "doc_id": "cache_" + str(hash(user_query)),
                    "text": user_query,
                    "file_name": "cache",
                    "updated_at": int(time.time())
                }]
            )
            print("Response cached for future queries.")

if __name__ == "__main__":
    main()