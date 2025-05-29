"""
API Server for Knowledge Base Access.

This module provides a RESTful API for uploading documents, 
managing the knowledge base, and performing semantic searches.
"""

import os
import logging
import time
import uuid
import argparse
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our knowledge base components
from knowledge_base import KnowledgeBaseConstructor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Knowledge Base API",
    description="API for document processing, vectorization, and semantic search in the enterprise knowledge base",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query", example="How does knowledge base help with decision making?")
    n_results: int = Field(5, description="Number of results to return", example=5, ge=1, le=20)
    filter_by: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria")

class SearchResult(BaseModel):
    text: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

class UploadResponse(BaseModel):
    file_name: str
    file_size: int
    segments_created: int
    processing_time_ms: float
    document_ids: List[str]
    success: bool
    message: str

class StatsResponse(BaseModel):
    documents_processed: int
    segments_created: int
    vectors_stored: int
    document_types: Dict[str, int]
    uptime_seconds: float

# Global variables
kb: Optional[KnowledgeBaseConstructor] = None
start_time = time.time()
upload_dir = "./uploaded_documents"
os.makedirs(upload_dir, exist_ok=True)

# Background tasks
def process_document_async(file_path: str, original_filename: str):
    """Process a document in the background."""
    try:
        logger.info(f"Background processing of {original_filename}")
        kb.add_document_to_kb(file_path, segment=True)
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {e}")

# API endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enterprise Knowledge Base API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": time.time() - start_time
    }

@app.post("/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    segment: bool = Form(True),
    async_processing: bool = Form(False)
):
    """Upload and process a document for the knowledge base."""
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    
    try:
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()
        file_path = os.path.join(upload_dir, f"{file_id}{file_ext}")
        
        # Read and save the file
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        file_size = len(file_content)
        logger.info(f"Uploaded file {original_filename} ({file_size} bytes) to {file_path}")
        
        # Process the file
        if async_processing:
            # Add to background tasks
            background_tasks.add_task(process_document_async, file_path, original_filename)
            
            return UploadResponse(
                file_name=original_filename,
                file_size=file_size,
                segments_created=0,  # Will be processed asynchronously
                processing_time_ms=0,
                document_ids=[],
                success=True,
                message="Document queued for processing"
            )
        else:
            # Process synchronously
            start_time = time.time()
            
            # Process and add the document
            document_ids = kb.add_document_to_kb(file_path, segment=segment)
            
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            
            segments_created = len(document_ids)
            
            return UploadResponse(
                file_name=original_filename,
                file_size=file_size,
                segments_created=segments_created,
                processing_time_ms=processing_time_ms,
                document_ids=document_ids,
                success=True,
                message=f"Document processed successfully into {segments_created} segments"
            )
            
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(search_request: SearchRequest):
    """Perform semantic search in the knowledge base."""
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    
    try:
        start_time = time.time()
        
        # Perform the search
        results = kb.search_kb(
            query=search_request.query,
            n_results=search_request.n_results,
            filter_by=search_request.filter_by
        )
        
        # Process the results
        search_results = []
        if results and 'documents' in results and len(results['documents']) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                relevance_score = 1.0 - distance  # Convert distance to relevance score
                
                search_results.append(SearchResult(
                    text=doc,
                    source=metadata.get('file_name', 'Unknown'),
                    relevance_score=relevance_score,
                    metadata=metadata
                ))
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=search_request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time * 1000
        )
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/stats", response_model=StatsResponse, tags=["Management"])
async def get_stats():
    """Get statistics about the knowledge base."""
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    
    try:
        kb_stats = kb.get_stats()
        
        return StatsResponse(
            documents_processed=kb_stats["documents_processed"],
            segments_created=kb_stats["segments_created"],
            vectors_stored=kb_stats["vectors_stored"],
            document_types=kb_stats["document_types"],
            uptime_seconds=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def start_server(host="0.0.0.0", port=8000, embedding_model="all-MiniLM-L6-v2", db_dir="./kb_api_db"):
    """Start the API server."""
    global kb
    
    # Initialize the knowledge base
    logger.info(f"Initializing knowledge base with model: {embedding_model}")
    kb = KnowledgeBaseConstructor(
        embedding_model_name=embedding_model,
        collection_name="api_kb",
        db_directory=db_dir
    )
    
    # Start the FastAPI server
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Base API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--db-dir", type=str, default="./kb_api_db", help="Directory for vector database")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        embedding_model=args.model,
        db_dir=args.db_dir
    )
