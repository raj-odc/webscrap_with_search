import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import openai  # Make sure to install the OpenAI library

# Initialize FAISS index
dimension = 512  # Adjust based on your vector size
index = faiss.IndexFlatL2(dimension)
stored_vectors = []  # To keep track of stored vectors
stored_texts = []    # To keep track of the original texts

# Set your OpenAI API key
openai.api_key = 'OPEN_API_KEY'  # Replace with your actual OpenAI API key

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs as an example
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        st.error("Failed to retrieve the website.")
        return None

def vectorize_text(text):
    # Dummy vectorization: Replace with your actual vectorization logic
    vector = np.random.rand(dimension).astype('float32')  # Random vector for demonstration
    return vector

def search_vectors(query):
    # Use OpenAI to process the search query
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate model
        messages=[{"role": "user", "content": query}]
    )
    
    # Extract the response text
    query_vector = response['choices'][0]['message']['content']
    
    # Convert the response text to a vector (you may need to implement your own vectorization logic here)
    query_vector = vectorize_text(query_vector)  # Replace with actual vectorization logic
    query_vector = np.array([query_vector]).astype('float32')

    # Search in the FAISS index
    distances, indices = index.search(query_vector, k=5)  # Get top 5 results
    results = []
    for i in indices[0]:
        if i >= 0 and i < len(stored_texts):  # Ensure index is within bounds
            results.append(stored_texts[i])
    return results

st.title("Website Scraper and Vector DB")

url = st.text_input("Enter the website URL:")

if st.button("Scrape and Store"):
    if url:
        scraped_data = scrape_website(url)
        if scraped_data:
            vector = vectorize_text(scraped_data)
            index.add(np.array([vector]))  # Add vector to FAISS index
            
            stored_vectors.append(vector)    # Store the vector
            print(stored_vectors)
            stored_texts.append(scraped_data) # Store the original text
            st.success("Data scraped and stored in vector DB!")
    else:
        st.warning("Please enter a valid URL.")

search_query = st.text_input("Search stored data:")
if search_query:
    search_results = search_vectors(search_query)
    if search_results:
        st.write("Search Results:")
        for result in search_results:
            st.write(result)
    else:
        st.write("No results found.")

