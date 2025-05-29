"""
Structured Data Processing Module for Knowledge Base Construction.

This module handles the loading and preprocessing of structured data formats
such as CSV, JSON, and XML for integration into the knowledge base.
"""

import os
import csv
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructuredDataProcessor:
    """Handles loading and preprocessing of structured data formats."""
    
    def __init__(self):
        """Initialize the structured data processor."""
        logger.info("Structured data processor initialized")
    
    def process_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a structured data file and convert it to text.
        
        Args:
            file_path: Path to the structured data file
            
        Returns:
            A tuple containing:
                - The text representation of the structured data
                - Metadata about the original data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Process based on file type
        if ext == '.json':
            return self._process_json(file_path)
        elif ext == '.csv':
            return self._process_csv(file_path)
        elif ext == '.xml':
            return self._process_xml(file_path)
        else:
            raise ValueError(f"Unsupported structured data format: {ext}")
    
    def _process_json(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a JSON file and convert it to text.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            A tuple containing the text representation and metadata
        """
        logger.info(f"Processing JSON file: {file_path}")
        
        try:
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Metadata
            metadata = {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_type': 'json',
                'structured_data': True
            }
            
            # Convert to textual representation
            text_representation = self._json_to_text(data)
            
            return text_representation, metadata
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            raise
    
    def _process_csv(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a CSV file and convert it to text.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            A tuple containing the text representation and metadata
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        try:
            # Read CSV file
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)
            
            # Extract headers and data
            if not rows:
                return "", {"file_name": os.path.basename(file_path)}
            
            headers = rows[0]
            data_rows = rows[1:]
            
            # Metadata
            metadata = {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_type': 'csv',
                'structured_data': True,
                'column_count': len(headers),
                'row_count': len(data_rows)
            }
            
            # Convert to text representation
            text_parts = []
            
            # Add table title/description
            table_name = os.path.basename(file_path).replace('.csv', '')
            text_parts.append(f"{table_name.upper()} DATA TABLE")
            text_parts.append(f"Contains {len(data_rows)} records with {len(headers)} columns.")
            text_parts.append("")
            
            # Add column definitions
            text_parts.append("COLUMNS:")
            for i, header in enumerate(headers):
                text_parts.append(f"- {header}: Column {i+1}")
            text_parts.append("")
            
            # Add data in a readable format
            text_parts.append("DATA RECORDS:")
            for i, row in enumerate(data_rows[:20]):  # Limit to first 20 rows for readability
                text_parts.append(f"Record {i+1}:")
                for j, cell in enumerate(row):
                    if j < len(headers):  # Ensure we don't go out of bounds
                        text_parts.append(f"  {headers[j]}: {cell}")
                text_parts.append("")
            
            # Indicate if there are more rows
            if len(data_rows) > 20:
                text_parts.append(f"... and {len(data_rows) - 20} more records.")
            
            text_representation = "\n".join(text_parts)
            
            return text_representation, metadata
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            raise
    
    def _process_xml(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process an XML file and convert it to text.
        
        Args:
            file_path: Path to the XML file
            
        Returns:
            A tuple containing the text representation and metadata
        """
        logger.info(f"Processing XML file: {file_path}")
        
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Metadata
            metadata = {
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_type': 'xml',
                'structured_data': True,
                'root_element': root.tag
            }
            
            # Convert to text representation
            text_parts = []
            
            # Add document title
            text_parts.append(f"XML DOCUMENT: {os.path.basename(file_path)}")
            text_parts.append(f"Root element: {root.tag}")
            text_parts.append("")
            
            # Convert XML to text with indentation
            self._xml_element_to_text(root, text_parts, indent_level=0)
            
            text_representation = "\n".join(text_parts)
            
            return text_representation, metadata
            
        except Exception as e:
            logger.error(f"Error processing XML file {file_path}: {e}")
            raise
    
    def _json_to_text(self, data: Any, parent_key: str = "ROOT", indent_level: int = 0) -> str:
        """
        Convert a JSON structure to a readable text representation.
        
        Args:
            data: The JSON data (dict, list, or primitive)
            parent_key: The key of the parent element
            indent_level: Current indentation level
            
        Returns:
            Text representation of the JSON data
        """
        indent = "  " * indent_level
        text_parts = []
        
        if isinstance(data, dict):
            # It's an object
            if indent_level == 0:
                text_parts.append(f"JSON DOCUMENT STRUCTURE:")
                text_parts.append("")
            
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    # Complex structure - add header and recurse
                    text_parts.append(f"{indent}- {key}:")
                    text_parts.append(self._json_to_text(value, key, indent_level + 1))
                else:
                    # Simple value
                    text_parts.append(f"{indent}- {key}: {value}")
            
        elif isinstance(data, list):
            # It's an array
            if not data:
                text_parts.append(f"{indent}Empty list")
            else:
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        # Complex item
                        if isinstance(item, dict) and "name" in item:
                            # Use name as identifier if available
                            item_name = f"{parent_key} item '{item['name']}'"
                        else:
                            # Use index otherwise
                            item_name = f"{parent_key} item #{i+1}"
                        
                        text_parts.append(f"{indent}- {item_name}:")
                        text_parts.append(self._json_to_text(item, parent_key, indent_level + 1))
                    else:
                        # Simple item
                        text_parts.append(f"{indent}- {item}")
        else:
            # It's a primitive value
            text_parts.append(f"{indent}{data}")
        
        return "\n".join(text_parts)
    
    def _xml_element_to_text(self, element: ET.Element, text_parts: List[str], indent_level: int = 0) -> None:
        """
        Convert an XML element to text representation.
        
        Args:
            element: The XML element
            text_parts: List to append text parts to
            indent_level: Current indentation level
        """
        indent = "  " * indent_level
        
        # Add element start
        attrs_text = ""
        if element.attrib:
            attrs_text = ", attributes: " + ", ".join([f"{k}='{v}'" for k, v in element.attrib.items()])
        
        text_parts.append(f"{indent}Element: {element.tag}{attrs_text}")
        
        # Add element text if present and meaningful
        if element.text and element.text.strip():
            text_parts.append(f"{indent}  Text: {element.text.strip()}")
        
        # Process child elements
        for child in element:
            self._xml_element_to_text(child, text_parts, indent_level + 1)


# Usage example
if __name__ == "__main__":
    processor = StructuredDataProcessor()
    
    # Example usage
    sample_files = {
        "json": "sample.json",
        "csv": "sample.csv",
        "xml": "sample.xml"
    }
    
    for format_name, file_path in sample_files.items():
        if os.path.exists(file_path):
            try:
                text, metadata = processor.process_file(file_path)
                print(f"\nProcessed {format_name.upper()} file: {file_path}")
                print(f"Metadata: {metadata}")
                print(f"First 200 chars: {text[:200]}...")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
