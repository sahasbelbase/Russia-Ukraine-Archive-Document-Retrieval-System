import streamlit as st
import requests

# Set the API endpoint
API_URL = "http://127.0.0.1:8000/search_document/"  # Adjust the URL as needed

# Streamlit UI components
st.title("Russia-Ukraine Archive: Document Retrieval System")

# Search box for the query
query_text = st.text_input("Enter your search query:")

# Button to trigger the search
if st.button("Search"):
    if query_text:
        # Prepare the payload for the API request
        payload = {"query_text": query_text}
        
        # Make the API request
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            results = response.json().get("documents", [])
            if results:
                st.subheader("Search Results")
                
                # Display the results in card format
                for result in results:
                    with st.expander(f"{result['rank']}. {result['document_name']}", expanded=True):
                        st.write(f"**Similarity Score:** {result['similarity_score']:.4f}")
            else:
                st.warning("No documents found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a query.")
