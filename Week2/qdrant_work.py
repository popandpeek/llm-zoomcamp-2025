from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient("http://localhost:6333")

def create_collection(collection, size):
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE)
    )

def upsert_to_collection(collection, vector, uid):
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=uid, vector=vector, payload={"label": "text"})
        ]
    )

def query_score(collection, query, limit=1):
    results = client.query_points(
        collection_name=collection,
        query=query,
        limit=limit
    )

    return results
