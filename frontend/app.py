import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search_document/" 


st.title("Russia-Ukraine Archive: Document Retrieval System")

query_text = st.text_input("Enter your search query:")

if st.button("Search"):
    if query_text:
        payload = {"query_text": query_text}
        
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            results = response.json().get("documents", [])
            if results:
                st.subheader("Search Results")

                for result in results:
                    with st.expander(f"{result['rank']}. {result['document_name']}", expanded=True):
                        st.write(f"**Similarity Score:** {result['similarity_score']:.4f}")
            else:
                st.warning("No documents found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a query.")
