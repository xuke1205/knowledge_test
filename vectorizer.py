"""
Vectorization and Vector Database Module for Knowledge Base Construction.

This module handles the embedding generation and vector storage of text segments
for efficient semantic search and retrieval.
"""

import os
import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Union, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextVectorizer:
    """Handles conversion of text to vector embeddings using embedding models."""
    
    def __init__(
        self, 
        model_name_or_path: str = "all-MiniLM-L6-v2", 
        device: Optional[str] = None
    ):
        """
        Initialize the text vectorizer.
        
        Args:
            model_name_or_path: The name or path of the sentence transformer model to use
            device: Device to use for computation ('cpu', 'cuda', etc.)
        """
        logger.info(f"Initializing TextVectorizer with model: {model_name_or_path}")
        
        # Load the embedding model
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        # Convert numpy arrays to list for JSON serializability
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Successfully generated {len(embeddings_list)} embeddings")
        return embeddings_list


class SentenceTransformerEmbeddingFunction:
    """Custom embedding function for ChromaDB using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize with a specific sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to use for computation ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Args:
            input: List of texts to embed
            
        Returns:
            List of embeddings as float lists
        """
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class VectorStore:
    """Handles storage and retrieval of text and associated embeddings in a vector database."""
    
    def __init__(
        self, 
        collection_name: str = "knowledge_base", 
        persist_directory: Optional[str] = "./chroma_db",
        embedding_function = None,
        tenant: str = "default_tenant",
        database: str = "default_database"
    ):
        """
        Initialize the vector storage.
        
        Args:
            collection_name: Name of the collection in the vector database
            persist_directory: Directory to persist the database (None for in-memory)
            embedding_function: Custom embedding function (if None, relies on pre-computed embeddings)
        """
        logger.info(f"Initializing VectorStore with collection: {collection_name}")
        
        # Initialize ChromaDB client
        settings = Settings()
        if persist_directory:
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory, 
                settings=settings,
                tenant=tenant,
                database=database
            )
        else:
            self.client = chromadb.EphemeralClient(
                settings=settings,
                tenant=tenant,
                database=database
            )
        
        # Create a default embedding function if none is provided
        self.embedding_function = embedding_function
        
        # Create or get collection
        self.collection_name = collection_name
        # 先尝试获取现有集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            # 如果不存在则创建
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Knowledge base collection"}
            )
            
        logger.info(f"Vector store initialized with collection: {collection_name}")
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            embeddings: Optional pre-computed embeddings (must match document order)
            batch_size: Number of documents to add in each batch
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        if not documents:
            return []
            
        # Generate document IDs if not present
        doc_ids = []
        for doc in documents:
            if 'id' not in doc:
                doc['id'] = str(uuid.uuid4())
            doc_ids.append(doc['id'])
        
        # Prepare document texts, metadata, and ids
        texts = [doc['text'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        ids = doc_ids
        
        # Add documents in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Use provided embeddings if available
            batch_embeddings = None
            if embeddings:
                batch_embeddings = embeddings[i:batch_end]
            
            # Add documents to collection
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
            
            logger.info(f"Added batch of {len(batch_ids)} documents ({i+1}-{batch_end}/{len(documents)})")
        
        logger.info(f"Successfully added {len(doc_ids)} documents to vector store")
        return doc_ids
    
    def search(
        self, 
        query_text: str, 
        n_results: int = 5, 
        query_embedding: Optional[List[float]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_text: The text query to search for
            n_results: Number of results to return
            query_embedding: Optional pre-computed embedding for the query
            where: Optional filter on metadata
            where_document: Optional filter on document content
            include: What to include in the results (e.g., ["metadatas", "documents", "distances"])
            
        Returns:
            Search results
        """
        logger.info(f"Searching for: '{query_text[:50]}...'")
        
        # Default include list if not specified
        include = include or ["metadatas", "documents", "distances"]
        
        # Perform the query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        logger.info(f"Found {len(results['documents'][0])} matching documents")
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "peek_sample": self.collection.peek(10) if count > 0 else None
        }
    
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None):
        """
        Delete documents from the collection.
        
        Args:
            ids: List of document IDs to delete
            where: Filter condition for documents to delete
        """
        if ids:
            logger.info(f"Deleting {len(ids)} documents by ID")
            self.collection.delete(ids=ids)
        elif where:
            logger.info(f"Deleting documents using filter: {where}")
            self.collection.delete(where=where)
        else:
            logger.warning("No deletion performed - must provide either ids or where")


class VectorizationPipeline:
    """
    End-to-end pipeline for vectorizing documents and storing in vector database.
    
    Combines the TextVectorizer and VectorStore into a single pipeline.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db",
        device: Optional[str] = None,
        tenant: str = "default_tenant",
        database: str = "default_database"
    ):
        """
        Initialize the vectorization pipeline.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            collection_name: Name of the vector store collection
            persist_directory: Directory to persist the vector store
            device: Device to use for embedding computation
        """
        logger.info(f"Initializing vectorization pipeline with model {embedding_model_name}")
        
        # Initialize vectorizer
        self.vectorizer = TextVectorizer(
            model_name_or_path=embedding_model_name,
            device=device
        )
        
        # Create embedding function for ChromaDB
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name,
            device=device
        )
        
        # Initialize vector store with the embedding function
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
            tenant=tenant,
            database=database
        )
        
        logger.info("Vectorization pipeline initialized")
    
    def process_documents(
        self, 
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[str]:
        """
        Process documents through the full pipeline: vectorize and store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs generated for the documents
        """
        logger.info(f"Processing {len(documents)} documents through vectorization pipeline")
        
        # Extract texts for vectorization
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.vectorizer.generate_embeddings(texts, batch_size=batch_size)
        
        # Store documents with embeddings
        doc_ids = self.vector_store.add_documents(documents, embeddings=embeddings)
        
        logger.info(f"Successfully processed {len(doc_ids)} documents")
        return doc_ids
    
    def search(
        self, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for documents using the vector store.
        
        Args:
            query_text: The query text
            n_results: Number of results to return
            where: Optional filter on metadata
            
        Returns:
            Search results
        """
        # Generate embedding for the query
        query_embedding = self.vectorizer.generate_embeddings([query_text])[0]
        
        # Search using the vector store
        results = self.vector_store.search(
            query_text=query_text,
            n_results=n_results,
            query_embedding=query_embedding,
            where=where
        )
        
        return results

# Usage example
if __name__ == "__main__":
    # Sample documents
    sample_documents = [
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "metadata": {"source": "encyclopedia", "topic": "technology"}
        },
        {
            "text": "Machine learning is a method of data analysis that automates analytical model building.",
            "metadata": {"source": "textbook", "topic": "technology"}
        },
        {
            "text": "Neural networks are computing systems with interconnected nodes that work much like neurons in the human brain.",
            "metadata": {"source": "research_paper", "topic": "computer_science"}
        },
        {
            "text": "Natural language processing (NLP) is a field of AI focused on the interaction between computers and humans using natural language.",
            "metadata": {"source": "article", "topic": "technology"}
        },
        {
            "text": "Computer vision is a field of AI that enables computers to derive meaningful information from digital images.",
            "metadata": {"source": "blog", "topic": "technology"}
        }
    ]
    
    # Initialize pipeline
    pipeline = VectorizationPipeline(
        embedding_model_name="all-MiniLM-L6-v2",
        collection_name="sample_collection",
        persist_directory="./sample_chroma_db"
    )
    
    # Process documents
    doc_ids = pipeline.process_documents(sample_documents)
    print(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")
    
    # Perform a search
    query = "How do neural networks work?"
    results = pipeline.search(query, n_results=2)
    
    print(f"\nSearch results for query: '{query}'")
    print("=" * 50)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"Result {i+1}:")
        print(f"- Text: {doc}")
        print(f"- Metadata: {metadata}")
        print(f"- Distance: {distance}")
        print("-" * 50)
