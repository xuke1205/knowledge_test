"""
Knowledge Base Construction Application.

This module brings together all components to demonstrate a complete
knowledge base construction pipeline from document processing to search.
"""

import os
import logging
import argparse
import json
import time
from typing import List, Dict, Any, Optional, Union

# Import our components - ensure there are no llama_index dependencies
from document_processor import DocumentProcessor
from structured_data_processor import StructuredDataProcessor
from segmenter import TextSegmenter, SemanticSegmenter  # Using our custom implementation without llama_index
from vectorizer import VectorizationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseConstructor:
    """
    End-to-end knowledge base construction system.
    
    Combines document processing, segmentation, and vectorization into a complete pipeline.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "enterprise_kb",
        db_directory: str = "./enterprise_kb_db", 
        device: Optional[str] = None,
        tenant: str = "default_tenant",
        database: str = "default_database"
    ):
        # 确保数据库目录存在
        if db_directory and not os.path.exists(db_directory):
            os.makedirs(db_directory, exist_ok=True)
        """
        Initialize the knowledge base constructor.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            collection_name: Name of the vector store collection
            db_directory: Directory to persist the vector database
            device: Device to use for computation
        """
        logger.info("Initializing Knowledge Base Constructor")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.structured_data_processor = StructuredDataProcessor()
        self.segmenter = TextSegmenter(chunk_size=512, chunk_overlap=50)
        
        # Initialize vectorization pipeline
        self.vectorization_pipeline = VectorizationPipeline(
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            persist_directory=db_directory,
            device=device,
            tenant=tenant,
            database=database
        )
        
        # Statistics tracking
        self.stats = {
            "documents_processed": 0,
            "segments_created": 0,
            "vectors_stored": 0,
            "document_types": {}
        }
        
        logger.info("Knowledge Base Constructor initialized")
    
    def process_document(self, file_path: str, segment: bool = True) -> List[Dict[str, Any]]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to the document file
            segment: Whether to segment the document (or keep as one chunk)
            
        Returns:
            List of processed document segments ready for vectorization
        """
        logger.info(f"Processing document: {file_path}")
        
        # Track document types
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        self.stats["document_types"][ext] = self.stats["document_types"].get(ext, 0) + 1
        
        # Step 1: Load and clean document based on file type
        if ext in ['.json', '.csv', '.xml', '.xlsx']:
            # Handle structured data formats
            try:
                cleaned_text, metadata = self.structured_data_processor.process_file(file_path)
                # Convert complex data types to string if needed
                if isinstance(cleaned_text, (dict, list)):
                    cleaned_text = json.dumps(cleaned_text, ensure_ascii=False)
                elif not isinstance(cleaned_text, str):
                    cleaned_text = str(cleaned_text)
                # Ensure Chinese text is properly encoded
                if isinstance(cleaned_text, str):
                    try:
                        cleaned_text = cleaned_text.encode('utf-8', errors='strict').decode('utf-8')
                    except UnicodeError:
                        try:
                            cleaned_text = cleaned_text.encode('gbk', errors='strict').decode('gbk')
                        except Exception:
                            cleaned_text = cleaned_text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error processing structured file {file_path}: {e}")
                raise
        else:
            # Handle document formats (PDF, DOCX, TXT, etc.)
            cleaned_text, metadata = self.document_processor.process_document(file_path)
        
        # Update stats
        self.stats["documents_processed"] += 1
        
        # Step 2: Segment the document if required
        if segment:
            segments = self.segmenter.segment_text(cleaned_text, metadata)
            self.stats["segments_created"] += len(segments)
            logger.info(f"Document segmented into {len(segments)} chunks")
            return segments
        else:
            # Return as a single document
            self.stats["segments_created"] += 1
            return [{
                "text": cleaned_text,
                "metadata": metadata
            }]
    
    def add_document_to_kb(self, file_path: str, segment: bool = True) -> List[str]:
        """
        Process a document and add it to the knowledge base.
        
        Args:
            file_path: Path to the document file
            segment: Whether to segment the document (or keep as one chunk)
            
        Returns:
            List of document IDs for the added segments
        """
        # Process the document
        document_segments = self.process_document(file_path, segment)
        
        # Vectorize and store
        doc_ids = self.vectorization_pipeline.process_documents(document_segments)
        
        # Update stats
        self.stats["vectors_stored"] += len(doc_ids)
        
        logger.info(f"Document {os.path.basename(file_path)} added to knowledge base")
        return doc_ids
    
    def batch_add_documents(self, file_paths: List[str], segment: bool = True) -> Dict[str, List[str]]:
        """
        Process and add multiple documents to the knowledge base.
        
        Args:
            file_paths: List of paths to document files
            segment: Whether to segment the documents
            
        Returns:
            Dictionary mapping file paths to lists of document IDs
        """
        results = {}
        
        for file_path in file_paths:
            try:
                doc_ids = self.add_document_to_kb(file_path, segment)
                results[file_path] = doc_ids
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results[file_path] = []
        
        return results
    
    def search_kb(self, query: str, n_results: int = 5, filter_by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the knowledge base.
        
        Args:
            query: The search query
            n_results: Number of results to return
            filter_by: Optional filter criteria
            
        Returns:
            Search results
        """
        return self.vectorization_pipeline.search(query, n_results, filter_by)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base construction."""
        return self.stats


def create_sample_files(data_dir: str) -> List[str]:
    """Create sample files for demonstration purposes."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a sample TXT file
    txt_path = os.path.join(data_dir, "sample_article.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("""
Knowledge Base Systems for Enterprise Applications

Introduction
Knowledge base systems are crucial tools for modern enterprises, enabling them to manage, 
organize, and retrieve vast amounts of information efficiently. These systems serve as 
centralized repositories of corporate knowledge, supporting decision-making processes, 
employee training, and customer service operations.

Key Components
1. Content Repository - The foundation of any knowledge base is its repository, where 
   information is stored in structured and unstructured formats.
2. Semantic Indexing - Advanced systems employ semantic understanding to categorize and 
   index information based on meaning rather than just keywords.
3. Search Functionality - Robust search capabilities allow users to quickly locate relevant 
   information through natural language queries.
4. User Interface - An intuitive interface enables easy navigation and access to knowledge.

Benefits for Enterprises
Implementing a comprehensive knowledge base system offers numerous advantages:
- Improved decision-making through access to accurate and timely information
- Enhanced employee productivity by reducing time spent searching for information
- Preservation of institutional knowledge despite employee turnover
- Consistent customer service through standardized information access
- Accelerated onboarding and training processes

Implementation Challenges
Despite their benefits, knowledge base systems present several challenges:
- Content maintenance and keeping information up-to-date
- Balancing security with accessibility
- Ensuring adoption across the organization
- Integrating with existing enterprise systems

Conclusion
As enterprises continue to generate and collect increasing volumes of data, effective 
knowledge management becomes essential for maintaining competitive advantage. Well-designed 
knowledge base systems provide the infrastructure necessary to transform raw information 
into actionable insights, supporting organizational goals and operational efficiency.
        """)
    
    logger.info(f"Created sample TXT file at {txt_path}")
    
    # Create a JSON file for structural data
    json_path = os.path.join(data_dir, "product_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "products": [
                {
                    "id": "PRD001",
                    "name": "Enterprise Knowledge Base Platform",
                    "description": "A comprehensive platform for building, managing and deploying enterprise knowledge bases with advanced semantic search capabilities.",
                    "features": [
                        "Multi-format document processing",
                        "Semantic segmentation",
                        "Vector-based retrieval",
                        "Real-time updates",
                        "User feedback integration"
                    ],
                    "pricing": {
                        "basic": "$500/month",
                        "professional": "$2000/month",
                        "enterprise": "Custom pricing"
                    }
                },
                {
                    "id": "PRD002",
                    "name": "Knowledge Analytics Suite",
                    "description": "Advanced analytics tools for monitoring knowledge base usage, identifying gaps, and optimizing content.",
                    "features": [
                        "Usage dashboards",
                        "Content gap analysis",
                        "Search performance metrics",
                        "User satisfaction tracking",
                        "ROI calculation"
                    ],
                    "pricing": {
                        "standard": "$300/month",
                        "premium": "$800/month"
                    }
                }
            ]
        }, f, indent=2)
    
    logger.info(f"Created sample JSON file at {json_path}")
    
    # Create a CSV file for tabular data
    csv_path = os.path.join(data_dir, "usage_statistics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("""date,department,queries,successful_searches,average_search_time_ms
2025-01-01,Engineering,1250,1050,342
2025-01-01,Customer Support,2890,2540,298
2025-01-01,Sales,980,820,315
2025-01-01,HR,420,380,287
2025-01-02,Engineering,1180,1020,328
2025-01-02,Customer Support,3010,2650,312
2025-01-02,Sales,1050,890,305
2025-01-02,HR,390,350,291
2025-01-03,Engineering,1320,1140,336
2025-01-03,Customer Support,2950,2630,304
2025-01-03,Sales,1100,950,298
2025-01-03,HR,410,370,285
""")
    
    logger.info(f"Created sample CSV file at {csv_path}")
    
    return [txt_path, json_path, csv_path]


def demo_knowledge_base(clean_start=True):
    """Demonstrate the knowledge base construction pipeline."""
    
    # Create data directory and sample files
    data_dir = os.path.join(os.getcwd(), "data")
    sample_files = create_sample_files(data_dir)
    
    # Delete existing database if clean start is requested
    db_dir = "./demo_kb_db"
    if clean_start and os.path.exists(db_dir):
        logger.info(f"Cleaning existing database: {db_dir}")
        import shutil
        try:
            shutil.rmtree(db_dir)
            logger.info("Database directory cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning database directory: {e}")
    
    # Initialize knowledge base constructor
    kb = KnowledgeBaseConstructor(
        embedding_model_name="all-MiniLM-L6-v2",
        collection_name="demo_kb",
        db_directory=db_dir
    )
    
    # Process and add documents to knowledge base
    logger.info("Adding documents to knowledge base...")
    start_time = time.time()
    
    doc_ids = kb.batch_add_documents(sample_files)
    
    processing_time = time.time() - start_time
    logger.info(f"Documents processed in {processing_time:.2f} seconds")
    
    # Display statistics
    stats = kb.get_stats()
    logger.info(f"Knowledge Base Stats: {json.dumps(stats, indent=2)}")
    
    # Demonstrate search
    logger.info("\nPerforming sample searches...")
    
    queries = [
        "What are the key components of a knowledge base system?",
        "How does a knowledge base help enterprise decision making?",
        "What is the pricing for the Enterprise Knowledge Base Platform?",
        "What are the usage statistics for the Customer Support department?"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        results = kb.search_kb(query, n_results=2)
        
        print("-" * 80)
        print(f"Results for: '{query}'")
        print("-" * 80)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"Result {i+1}:")
            print(f"- Text: {doc[:200]}...")
            print(f"- Source: {metadata.get('file_name', 'Unknown')}")
            print(f"- Relevance score: {1.0 - distance:.4f}")
            print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Base Construction Demo")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for sample data files")
    parser.add_argument("--keep-db", action="store_true", help="Keep existing database (don't clean)")
    args = parser.parse_args()
    
    # Run the demonstration
    try:
        demo_knowledge_base(clean_start=not args.keep_db)
        print("Knowledge base demonstration completed successfully!")
    except Exception as e:
        logger.error(f"Error during knowledge base demonstration: {e}")
        print(f"ERROR: {e}")
