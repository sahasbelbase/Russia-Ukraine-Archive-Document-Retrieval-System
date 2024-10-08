import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')

app = FastAPI()

# Store documents and TF-IDF vectors
document_store = {}
tfidf_vectors_store = {}

class QueryModel(BaseModel):
    query_text: str

def process_document_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r'\w+', text.lower())
    return [token for token in tokens if token not in stop_words]

def load_documents_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Assuming documents are text files
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                document_text = file.read()
                document_store[filename] = document_text

def compute_tf(doc_tokens):
    tf = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1
    # Normalize TF
    total_tokens = len(doc_tokens)
    for token in tf:
        tf[token] /= total_tokens
    return tf

def compute_idf(documents):
    idf = {}
    total_documents = len(documents)
    for doc_tokens in documents:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            idf[token] = idf.get(token, 0) + 1
    # Calculate IDF
    for token in idf:
        idf[token] = np.log(total_documents / (idf[token] + 1))  
    return idf

def generate_tfidf_vector(query_tokens, processed_docs, word_vocab):
    tf = compute_tf(query_tokens)
    idf = compute_idf(processed_docs)
    
    tfidf_vector = np.zeros(len(word_vocab))
    for i, word in enumerate(word_vocab):
        tfidf_vector[i] = tf.get(word, 0) * idf.get(word, 0)
    return tfidf_vector

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

@app.on_event("startup")
def startup_event():
    load_documents_from_folder("C:\drive d\Drive C\Desktop\Meterials\document-retrieval-system\documents")  # Set your document folder path here
    global processed_docs
    processed_docs = [process_document_text(doc) for doc in document_store.values()]

@app.post("/search_document/")
async def search_document(query: QueryModel):
    if not document_store:
        raise HTTPException(status_code=404, detail="No documents processed. Please check the backend configuration.")

    query_tokens = process_document_text(query.query_text)
    
    word_vocab = sorted(set(word for doc in processed_docs for word in doc))
    
    query_vector = generate_tfidf_vector(query_tokens, processed_docs, word_vocab)

    similarity_scores = {
        filename: cosine_similarity(query_vector, generate_tfidf_vector(processed_docs[i], processed_docs, word_vocab))
        for i, filename in enumerate(document_store.keys())
    }
    sorted_documents = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    response_docs = [
        {"rank": index + 1, "document_name": doc[0], "similarity_score": doc[1]} 
        for index, doc in enumerate(sorted_documents)
    ]

    return {"documents": response_docs}
