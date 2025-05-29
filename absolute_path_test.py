"""
Test that writes output to a specific absolute path.
"""

import os
import sys
import datetime

# Define a more specific output path
output_file = "d:/mark重要资料/G-SQL/workdata/DATA SQL/test_results.txt"

try:
    # Current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Basic info
    info = [
        f"Test run at: {current_time}",
        f"Python version: {sys.version}",
        f"Current directory: {os.getcwd()}",
        f"Script location: {os.path.abspath(__file__)}",
        "\nAttempting to import modules...",
    ]
    
    # Try importing modules
    module_results = []
    
    # Test segmenter
    try:
        import segmenter
        module_results.append("✓ Segmenter module imported successfully")
    except Exception as e:
        module_results.append(f"✗ Error importing segmenter: {e}")
    
    # Test knowledge_base
    try:
        import knowledge_base
        module_results.append("✓ Knowledge base module imported successfully")
    except Exception as e:
        module_results.append(f"✗ Error importing knowledge_base: {e}")
        
    # Print results to console
    print("\n".join(info))
    print("\n".join(module_results))
    
    # Write results to the absolute path file
    print(f"\nWriting results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
        f.write("\n".join(module_results) + "\n")
    
    print(f"Results written successfully to {output_file}")
    
except Exception as e:
    print(f"ERROR in test script: {e}")

print("Test complete")
