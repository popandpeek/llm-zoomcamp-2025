import requests
import tiktoken
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import openai
import config

es = Elasticsearch(
        hosts=["http://localhost:9200"],
        # Optional: Add authentication if needed
        # basic_auth=("your_username", "your_password")
    )

first_question = 'How do execute a command on a Kubernetes pod?'
second_question = 'How do copy a file to a Docker container?'

documents = []


def create_index(index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)

def init_docs():
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

def init():
    init_docs()

    create_index("docs")

    actions = [
        {
            "_index": "docs",
            "_source": doc
        }
        for doc in documents
    ]

    bulk(es, actions)

def test_es_connection():
    if es.ping():
        print("Successfully connected to Elasticsearch")
    else:
        print("Could not connect to Elasticsearch")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #init()
    es.indices.refresh(index="docs")

    print("Length of documents: ", len(documents))

    # Q.1 - version.build.hash: "c6b8d8d951c631db715485edc1a74190cdce4189"
    info = es.info()
    build_hash = info['version']['build_hash']
    print(f"Q1. version.build.hash: {build_hash}")

    # Q.2: Index
    print(f"Q2. the 'index' function adds data to elasticsearch")

    first_query = {
        "query": {
            "multi_match": {
                "query": first_question,
                "type": "best_fields",
                "fields": ["question^4", "text"]
            }
        }
    }

    second_query = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": second_question,
                        "type": "best_fields",
                        "fields": ["question^4", "text"]
                    }
                },
                "filter": {
                    "term": {
                        "course.keyword": 'machine-learning-zoomcamp'   # Use .keyword for exact match on keyword field
                    }
                }
            }
        }
    }

    # Q.3 - 44.5055
    response = es.search(index='docs', query=first_query['query'])
    score = response["hits"]["hits"][0]["_score"]
    print(f"Q3. Score: {score}")

    # Q.4 - How do I copy files from a different folder into docker container’s working directory?
    response = es.search(index='docs', query=second_query['query'])
    third_question = response["hits"]["hits"][:3]
    print(f"Q4. Question: {third_question[2]["_source"]["question"]}")

    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    context = "\n\n".join(
        context_template.format(
            question=doc["_source"]["question"],
            text=doc["_source"]["text"]
        ) for doc in third_question
    )

    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    full_prompt = prompt_template.format(
        question="How do I execute a command in a running docker container?",
        context=context
    )

    # Q.5 - 1490
    print(f"Q5. Prompt length: {len(full_prompt)}")

    # Q.6 - 329
    encoding = tiktoken.encoding_for_model("gpt-4o")
    print(f"Q6. Tokens: {len(encoding.encode(full_prompt))}")