"""
Semantic Segmentation Module for Knowledge Base Construction.

This module handles the segmentation of documents into semantic chunks for better 
retrieval and understanding in the knowledge base.
"""

import logging
import textwrap
import re
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextSegmenter:
    """Handles semantic segmentation of text documents for knowledge base construction."""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        paragraph_separator: str = "\n\n",
        sentence_separator: str = ". "
    ):
        """
        Initialize the text segmenter.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            paragraph_separator: String that separates paragraphs
            sentence_separator: String that separates sentences
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.sentence_separator = sentence_separator
        
        logger.info(f"Text segmenter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # First split by paragraph
        paragraphs = re.split(re.escape(self.paragraph_separator), text)
        paragraphs = [p for p in paragraphs if p.strip()]
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        
        # Process each paragraph
        for paragraph in paragraphs:
            # If the paragraph itself is too large, split by sentence
            if len(paragraph) > self.chunk_size:
                sentences = re.split(re.escape(self.sentence_separator), paragraph)
                sentences = [s + self.sentence_separator for s in sentences if s.strip()]
                if not sentences[-1].endswith(self.sentence_separator):
                    sentences[-1] = sentences[-1].rstrip(self.sentence_separator)
                
                # Process each sentence
                for sentence in sentences:
                    # If adding this sentence would exceed chunk size, start a new chunk
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Add overlap by keeping some of the last text
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                    
                    current_chunk += sentence
            else:
                # If adding this paragraph would exceed chunk size, start a new chunk
                if len(current_chunk) + len(paragraph) + len(self.paragraph_separator) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap by keeping some of the last text
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                
                current_chunk += paragraph + self.paragraph_separator
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def segment_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Segment text into chunks based on semantic boundaries.
        
        Args:
            text: The text to segment
            metadata: Additional metadata to include with each segment
            
        Returns:
            List of dictionaries, each containing a text segment and its metadata
        """
        logger.info(f"Segmenting text of length {len(text)}")
        
        # Split the text
        chunks = self._split_text(text)
        
        # Create segments with metadata
        segments = []
        for i, chunk in enumerate(chunks):
            segment_metadata = {
                'segment_id': i,
                'segment_total': len(chunks)
            }
            
            if metadata:
                segment_metadata.update(metadata)
            
            segments.append({
                'text': chunk,
                'metadata': segment_metadata
            })
        
        logger.info(f"Text segmented into {len(segments)} chunks")
        return segments
    
    def preview_segmentation(self, text: str, max_preview: int = 5) -> None:
        """
        Preview the segmentation of a text (for debugging/analysis).
        
        Args:
            text: The text to segment
            max_preview: Maximum number of segments to preview
        """
        segments = self.segment_text(text)
        
        print(f"Text segmented into {len(segments)} chunks")
        print("=" * 50)
        
        for i, segment in enumerate(segments[:max_preview]):
            print(f"Segment {i+1}/{len(segments)}:")
            print("-" * 50)
            print(textwrap.fill(segment['text'][:200] + "..." if len(segment['text']) > 200 else segment['text'], width=80))
            print("-" * 50)
            
        if len(segments) > max_preview:
            print(f"... and {len(segments) - max_preview} more segments")


class SemanticSegmenter:
    """
    More advanced segmenter that uses semantic understanding to create better segments.
    
    This is a simplified version that doesn't rely on external libraries.
    In a production environment, this would use embedding models to determine
    semantic boundaries.
    """
    
    def __init__(
        self, 
        embedding_model=None,  # Placeholder for future implementation
        buffer_size: int = 3,
        breakpoint_percentile_threshold: int = 95
    ):
        """
        Initialize the semantic segmenter.
        
        Args:
            embedding_model: The embedding model to use for semantic analysis (placeholder)
            buffer_size: Buffer size for context overlap
            breakpoint_percentile_threshold: Threshold for semantic breakpoints
        """
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.embedding_model = embedding_model
        
        # Use TextSegmenter as a fallback
        self.basic_segmenter = TextSegmenter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info("Semantic segmenter initialized (using fallback basic segmenter for demo)")
        
    def segment_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Segment text using semantic understanding.
        
        Args:
            text: The text to segment
            metadata: Additional metadata to include with each segment
            
        Returns:
            List of dictionaries, each containing a text segment and its metadata
        """
        # For now, use the basic text segmenter as a fallback
        segments = self.basic_segmenter.segment_text(text, metadata)
        
        # Add semantic segmentation flag to metadata
        for segment in segments:
            segment['metadata']['semantic_segmentation'] = True
        
        logger.info(f"Text semantically segmented into {len(segments)} chunks (using fallback method)")
        return segments

# Usage example
if __name__ == "__main__":
    sample_text = """
    Knowledge management is the process of creating, sharing, using and managing 
    the knowledge and information of an organization. It refers to a multidisciplinary 
    approach to achieve organizational objectives by making the best use of knowledge.
    
    An established discipline since 1991, knowledge management includes courses taught 
    in the fields of business administration, information systems, management, library, 
    and information sciences. Other fields may contribute to knowledge management 
    research, including information and media, computer science, public health and 
    public policy.
    
    Several universities offer dedicated master's degrees in knowledge management.
    
    Many large companies, public institutions, and non-profit organizations have 
    resources dedicated to internal knowledge management efforts, often as a part 
    of their business strategy, IT, or human resource management departments.
    
    Knowledge management efforts typically focus on organizational objectives such 
    as improved performance, competitive advantage, innovation, the sharing of 
    lessons learned, integration, and continuous improvement of the organization.
    
    These efforts overlap with organizational learning and may be distinguished from 
    that by a greater focus on the management of knowledge as a strategic asset and 
    on encouraging the sharing of knowledge. Knowledge management is an enabler of 
    organizational learning.
    """
    
    # Basic text segmentation
    print("Basic Text Segmentation:")
    basic_segmenter = TextSegmenter(chunk_size=200, chunk_overlap=20)
    basic_segmenter.preview_segmentation(sample_text)
    
    print("\n" + "=" * 50 + "\n")
    
    # Semantic segmentation (using fallback splitter for demo)
    print("Semantic Segmentation (demo implementation):")
    semantic_segmenter = SemanticSegmenter()
    segments = semantic_segmenter.segment_text(sample_text)
    
    for i, segment in enumerate(segments[:3]):
        print(f"Segment {i+1}/{len(segments)}:")
        print("-" * 50)
        print(textwrap.fill(segment['text'][:200] + "..." if len(segment['text']) > 200 else segment['text'], width=80))
        print("-" * 50)
