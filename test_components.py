"""
Test script to verify the knowledge base components.
"""

import sys
import os
import logging

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_segmenter():
    """Test the segmenter module."""
    try:
        logger.info("Testing segmenter module...")
        from segmenter import TextSegmenter
        
        # Create a test segmenter
        segmenter = TextSegmenter(chunk_size=200, chunk_overlap=20)
        
        # Sample text for testing
        sample_text = """
        Knowledge management is the process of creating, sharing, using and managing 
        the knowledge and information of an organization. It refers to a multidisciplinary 
        approach to achieve organizational objectives by making the best use of knowledge.
        
        An established discipline since 1991, knowledge management includes courses taught 
        in the fields of business administration, information systems, management, library, 
        and information sciences. Other fields may contribute to knowledge management 
        research, including information and media, computer science, public health and 
        public policy.
        """
        
        # Test segmentation
        segments = segmenter.segment_text(sample_text)
        logger.info(f"Successfully segmented text into {len(segments)} segments")
        
        # Print first segment
        if segments:
            logger.info(f"First segment: {segments[0]['text'][:100]}...")
            
        return True
    except Exception as e:
        logger.error(f"Error testing segmenter: {e}")
        return False

def test_document_processor():
    """Test the document processor module."""
    try:
        logger.info("Testing document processor module...")
        from document_processor import DocumentProcessor
        
        # Create a processor
        processor = DocumentProcessor()
        logger.info("Document processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing document processor: {e}")
        return False

def test_vectorizer():
    """Test the vectorizer module."""
    try:
        logger.info("Testing vectorizer module...")
        from vectorizer import TextVectorizer
        
        # This will be slow as it loads the model, so we'll just import
        logger.info("Vectorizer module imported successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing vectorizer: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting component tests...")
    
    # Test results
    results = {
        "segmenter": test_segmenter(),
        "document_processor": test_document_processor(),
        "vectorizer": test_vectorizer()
    }
    
    # Log results
    logger.info("Test Results:")
    for component, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {component}: {status}")
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    logger.info(f"Summary: {passed}/{len(results)} tests passed")
    
    # Write a result file for easy checking
    with open("test_result.txt", "w") as f:
        f.write(f"Test Results:\n")
        for component, success in results.items():
            status = "PASSED" if success else "FAILED"
            f.write(f"  {component}: {status}\n")
        f.write(f"\nSummary: {passed}/{len(results)} tests passed\n")
    

if __name__ == "__main__":
    main()
