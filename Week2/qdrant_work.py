from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests

# docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
# docs_response = requests.get(docs_url)
# documents_raw = docs_response.json()
# course_documents = []
#
# for course in documents_raw:
#     course_name = course['course']
#     if course_name != 'machine-learning-zoomcamp':
#         continue
#
#     for doc in course['documents']:
#         doc['course'] = course_name
#         course_documents.append(doc)

client = QdrantClient("http://localhost:6333")

if client.collection_exists("test_collection"):
    client.delete_collection("test_collection")

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE)
)

client.upsert(
    collection_name="test_collection",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"label": "first"})
    ]
)

results = client.query_points(
    collection_name="test_collection",
    query=[0.1, 0.2, 0.3, 0.4],
    limit=1
)

print(results)
