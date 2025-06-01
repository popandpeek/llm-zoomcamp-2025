import requests
from elasticsearch import Elasticsearch

is_init = False
documents = []

def init_docs():
    global is_init
    is_init = True
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

def test_es_connection():
    es = Elasticsearch(
        hosts=["http://localhost:9200"],
        # Optional: Add authentication if needed
        # basic_auth=("your_username", "your_password")
    )
    if es.ping():
        print("Successfully connected to Elasticsearch")
    else:
        print("Could not connect to Elasticsearch")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_es_connection()


