import numpy as np
import sys
import os

def convert_to_npy(input_file, output_file):
    """Convert detection data to NPY format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Load data based on input file type
        if input_file.endswith('.txt'):
            # Load detections with mixed data types
            data = np.genfromtxt(input_file, delimiter=',', dtype=float, filling_values=-1)
        elif input_file.endswith('.csv'):
            data = np.genfromtxt(input_file, delimiter=',', dtype=float)
        else:
            raise ValueError("Unsupported input file format")
        
        # Save as NPY
        np.save(output_file, data)
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_npy.py input_file output_file.npy")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not output_file.endswith('.npy'):
        output_file += '.npy'
        
    convert_to_npy(input_file, output_file)