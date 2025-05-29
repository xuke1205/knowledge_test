"""
Streamlit Web Interface for Knowledge Base Management.

This module provides a user-friendly web interface for uploading documents,
managing the knowledge base, and performing semantic searches.
"""

import os
import time
import json
import logging
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = "/upload"
SEARCH_ENDPOINT = "/search"
STATS_ENDPOINT = "/stats"

# Set page configuration
st.set_page_config(
    page_title="Enterprise Knowledge Base Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for configuration and statistics
st.sidebar.title("Knowledge Base Dashboard")
st.sidebar.markdown("---")

# API Configuration
api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)

# Check API connection
def check_api_connection():
    try:
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.sidebar.success(f"Connected to API (v{data.get('version', 'unknown')})")
            st.sidebar.text(f"Uptime: {data.get('uptime_seconds', 0):.1f} seconds")
            return True
        else:
            st.sidebar.error(f"API returned status code {response.status_code}")
            return False
    except Exception as e:
        st.sidebar.error(f"Cannot connect to API: {e}")
        return False

# Get knowledge base stats
def get_kb_stats():
    try:
        response = requests.get(f"{api_url}{STATS_ENDPOINT}")
        if response.status_code == 200:
            return response.json()
        else:
            st.sidebar.warning(f"Failed to get stats: {response.status_code}")
            return None
    except Exception as e:
        st.sidebar.warning(f"Error fetching stats: {e}")
        return None

# Show statistics in sidebar
if check_api_connection():
    stats = get_kb_stats()
    if stats:
        st.sidebar.markdown("## Knowledge Base Statistics")
        st.sidebar.text(f"Documents processed: {stats['documents_processed']}")
        st.sidebar.text(f"Segments created: {stats['segments_created']}")
        st.sidebar.text(f"Vectors stored: {stats['vectors_stored']}")
        
        # Document types breakdown
        if stats['document_types']:
            st.sidebar.markdown("### Document Types")
            for doc_type, count in stats['document_types'].items():
                st.sidebar.text(f"{doc_type}: {count}")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard allows you to manage and interact with your enterprise "
    "knowledge base. Upload documents, search for information, and view statistics."
)

# Main content area
st.title("Enterprise Knowledge Base")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Search Knowledge Base", "Upload Documents"])

# Tab 1: Search Knowledge Base
with tab1:
    st.header("Semantic Search")
    st.markdown(
        "Enter your query to search the knowledge base. The system will find the most "
        "relevant information based on semantic similarity."
    )
    
    # Search form
    query = st.text_input("Enter your search query", key="search_query")
    col1, col2 = st.columns(2)
    num_results = col1.number_input("Number of results", min_value=1, max_value=20, value=5)
    search_button = col2.button("Search", key="search_button", use_container_width=True)
    
    # Execute search
    if search_button and query:
        try:
            with st.spinner("Searching..."):
                # Prepare the search request
                search_data = {
                    "query": query,
                    "n_results": num_results,
                    "filter_by": None
                }
                
                # Make the API request
                response = requests.post(
                    f"{api_url}{SEARCH_ENDPOINT}",
                    json=search_data
                )
                
                # Process and display results
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Found {results['total_results']} results in {results['search_time_ms']:.2f} ms")
                    
                    # Display each result
                    for i, result in enumerate(results['results']):
                        with st.expander(f"Result {i+1}: {result['source']} (Score: {result['relevance_score']:.4f})", expanded=i==0):
                            st.markdown(f"**Text:**\n{result['text']}")
                            st.markdown("**Metadata:**")
                            st.json(result['metadata'])
                else:
                    st.error(f"Error: {response.status_code}\n{response.text}")
            
        except Exception as e:
            st.error(f"Search failed: {e}")

# Tab 2: Upload Documents
with tab2:
    st.header("Upload Documents")
    st.markdown(
        "Upload documents to add to the knowledge base. Supported formats include PDF, DOCX, TXT, "
        "CSV, JSON, and XML."
    )
    
    # Upload form
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "json", "xml"])
    
    col1, col2 = st.columns(2)
    segment = col1.checkbox("Segment document", value=True)
    async_processing = col2.checkbox("Process asynchronously", value=False)
    
    upload_button = st.button("Upload to Knowledge Base", key="upload_button", use_container_width=True, 
                             disabled=uploaded_file is None)
    
    # Process upload
    if upload_button and uploaded_file:
        try:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Prepare form data
                files = {"file": (uploaded_file.name, open(temp_file_path, "rb"))}
                form_data = {
                    "segment": "true" if segment else "false",
                    "async_processing": "true" if async_processing else "false"
                }
                
                # Make the API request
                response = requests.post(
                    f"{api_url}{UPLOAD_ENDPOINT}",
                    files=files,
                    data=form_data
                )
                
                # Clean up
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                # Display result
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Document uploaded successfully: {result['file_name']}")
                    
                    result_display = {
                        "File Name": result['file_name'],
                        "File Size": f"{result['file_size']} bytes",
                        "Segments Created": result['segments_created'],
                        "Processing Time": f"{result['processing_time_ms']:.2f} ms",
                        "Message": result['message']
                    }
                    
                    # Display result in a table
                    st.json(result_display)
                    
                    if async_processing:
                        st.info("Document is being processed in the background. Check statistics for updates.")
                else:
                    st.error(f"Error: {response.status_code}\n{response.text}")
                    
        except Exception as e:
            st.error(f"Upload failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    "Enterprise Knowledge Base System - MVP Proof of Concept | "
    "Built with Streamlit, FastAPI, ChromaDB, and sentence-transformers"
)


if __name__ == "__main__":
    # This app is run using 'streamlit run streamlit_app.py'
    pass
