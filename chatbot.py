import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import time
import sklearn

class WebsiteChatbot:
    def __init__(self):
        self.url = None
        self.content = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def scrape_website(self, url):
        """Scrape text content from the website"""
        self.url = url
        try:
            with st.spinner('Scraping website content...'):
                # Send GET request to the URL
                response = requests.get(self.url)
                response.raise_for_status()
                
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text from paragraphs and headers
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                
                # Clear previous content
                self.content = []
                
                # Process and store each text segment
                for element in text_elements:
                    text = element.get_text().strip()
                    if text and len(text) > 20:  # Only keep meaningful segments
                        self.content.append(text)
                
                # Create TF-IDF matrix
                self.tfidf_matrix = self.vectorizer.fit_transform(self.content)
                
                return len(self.content)
                
        except Exception as e:
            st.error(f"Error scraping website: {str(e)}")
            return 0
    
    def get_response(self, user_question, top_k=3):
        """Generate a response based on the user's question"""
        if not self.content:
            return "I don't have any content to work with. Please scrape a website first."
        
        # Transform user question using the same vectorizer
        question_vector = self.vectorizer.transform([user_question])
        
        # Calculate similarity between question and all content
        similarities = cosine_similarity(question_vector, self.tfidf_matrix)
        
        # Debug: Print similarities
        print("Similarities:", similarities)

        # Get indices of top k most similar content pieces
        top_indices = similarities[0].argsort()[-top_k:][::-1]
        
        # If best similarity is too low, return a default response
        if similarities[0][top_indices[0]] < 0.3:  # Adjust this value
            return "I'm sorry, I couldn't find relevant information to answer your question."
        
        # Construct response from most relevant content
        response = "Based on the website content:"
        for idx in top_indices:
            similarity_score = similarities[0][idx] * 100  # Use the correct index
            response += f"\n\n• {self.content[idx]}\n(Relevance: {similarity_score:.1f}%)"
        
        return response.strip()

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebsiteChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'url' not in st.session_state:
        st.session_state.url = ""

def main():
    st.set_page_config(page_title="Website Chatbot", page_icon="🤖", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for website URL input
    with st.sidebar:
        st.title("Website Chatbot Settings")
        url_input = st.text_input("Enter website URL:", value=st.session_state.url)
        if st.button("Scrape Website"):
            if url_input:
                num_segments = st.session_state.chatbot.scrape_website(url_input)
                if num_segments > 0:
                    st.session_state.url = url_input
                    st.success(f"Successfully scraped {num_segments} text segments!")
                    st.session_state.messages = []  # Clear chat history for new website
            else:
                st.error("Please enter a valid URL")
    
    # Main chat interface
    st.title("Website Content Chatbot 🤖")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a question about the website content"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_response(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display initial message if no website is scraped
    if not st.session_state.url:
        st.info("👈 Please enter a website URL in the sidebar and click 'Scrape Website' to begin.")

if __name__ == "__main__":
    print(sklearn.__version__)
    main()