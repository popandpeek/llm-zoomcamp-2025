from fastembed import TextEmbedding
import numpy as np
import requests
from qdrant_work import create_collection, upsert_to_collection

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
course_documents = []

for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue

    for doc in course['documents']:
        doc['course'] = course_name
        course_documents.append(doc)

embedding_model = TextEmbedding(model_name='jinaai/jina-embeddings-v2-small-en')
second_embedding_model = TextEmbedding(model_name='BAAI/bge-small-en')
initial_documents = ['I just discovered the course. Can I join now?', 'Can I still join the course after the start date?']
documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]


def embed_string(document):
    embeddings = embedding_model.embed(document)
    return list(embeddings)

if __name__ == '__main__':
    embedded_docs = embed_string(initial_documents)

    # Q1. - Embedding the query
    first_doc = embedded_docs[0].tolist()
    print('Q1. - ', min(first_doc))

    # Q2. - Cosine similarity with another vector
    second_doc = embedded_docs[1].tolist()
    print('Q2. - ', np.dot(first_doc, second_doc))

    # Q3. - Ranking by cosine
    text_embeddings = list(embedding_model.embed([doc['text'] for doc in documents]))
    similarities = [np.dot(embedded_docs[0].tolist(), text_embed) for text_embed in text_embeddings]
    best_idx = max(enumerate(similarities), key=lambda x: x[1])
    print('Q3. - ', best_idx[0], ':', best_idx[1])

    # Q4. - Ranking by cosine, version two
    full_text_embeddings = list(embedding_model.embed([doc['question'] + ' ' + doc['text'] for doc in documents]))
    ft_similarities = [np.dot(embedded_docs[0].tolist(), text_embed) for text_embed in full_text_embeddings]
    best_full_idx = max(enumerate(ft_similarities), key=lambda x: x[1])
    print('Q4. - ', best_full_idx[0], ':', best_full_idx[1])
    print('      ', 'Queries are included in question 4 data. The similarity is different - and higher - '
                    'due to the greater amount of information to compare.')

    #Q5. - Selecting the embedding model
    len_embedding_model = len(list(embedding_model.embed(['Test']))[0])
    len_second_embedding_model = len(list(second_embedding_model.embed(['Test']))[0])
    print('Q5. - model = jinaai/jina-embeddings-v2-small-en', ':', len_embedding_model)
    print('      model = BAAI/bge-small-en                 ', ':', len_second_embedding_model)

    #Q6. - Indexing with qdrant -> see qdrant_work.py
    course_embeddings = list(second_embedding_model.embed([doc['question'] + ' ' + doc['text'] for doc in course_documents]))
    create_collection('course_documents', len_second_embedding_model)
    for idx, e in enumerate(course_embeddings):
        upsert_to_collection('course_documents', e, idx)
    base_embedding = list(second_embedding_model.embed(initial_documents))
    cd_similarities = [np.dot(base_embedding[0].tolist(), text_embed) for text_embed in course_embeddings]
    print('Q6. - Highest similarity: ', max(cd_similarities))