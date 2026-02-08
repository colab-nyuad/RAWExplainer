"""
Compare Two Explanation Files

This script compares two explanation files to identify differences in structure,
content, and statistics. Useful when the same script generates different outputs.

Usage:
    python compare_explanation_files.py <file1.pt> <file2.pt>
"""

import torch
import sys
import os
import numpy as np


def load_and_describe(filepath):
    """Load a .pt file and return its structure info"""
    print(f"\nLoading: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
    
    data = torch.load(filepath, map_location='cpu')
    
    info = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'data': data,
        'type': type(data).__name__,
    }
    
    # Analyze based on type
    if isinstance(data, torch.Tensor):
        info['shape'] = data.shape
        info['dtype'] = str(data.dtype)
        info['size_mb'] = data.element_size() * data.numel() / (1024 * 1024)
        info['num_elements'] = data.numel()
        info['min'] = data.min().item() if data.numel() > 0 else None
        info['max'] = data.max().item() if data.numel() > 0 else None
        
        if len(data.shape) == 2:
            info['num_samples'] = data.shape[0]
            info['features_per_sample'] = data.shape[1]
            
            # Count padding (-1 values)
            if data.dtype in [torch.int32, torch.int64, torch.long]:
                num_padding = (data == -1).sum().item()
                info['num_padding'] = num_padding
                info['padding_pct'] = (num_padding / data.numel() * 100) if data.numel() > 0 else 0
                
                # Non-padding elements
                valid_data = data[data != -1]
                if valid_data.numel() > 0:
                    info['valid_min'] = valid_data.min().item()
                    info['valid_max'] = valid_data.max().item()
                    info['num_unique_values'] = len(torch.unique(valid_data))
                    info['valid_mean'] = valid_data.float().mean().item()
    
    elif isinstance(data, dict):
        info['keys'] = list(data.keys())
        info['num_keys'] = len(data.keys())
        info['key_details'] = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                info['key_details'][key] = {
                    'type': 'list',
                    'length': len(value),
                    'sample': value[:3] if len(value) > 0 else []
                }
            elif isinstance(value, torch.Tensor):
                info['key_details'][key] = {
                    'type': 'tensor',
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                info['key_details'][key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]
                }
    
    elif isinstance(data, list):
        info['length'] = len(data)
        info['first_element_type'] = type(data[0]).__name__ if len(data) > 0 else None
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            info['first_element_shape'] = data[0].shape
    
    return info


def print_info(info, indent=0):
    """Pretty print file info"""
    ind = "  " * indent
    
    print(f"{ind}File: {info['filename']}")
    print(f"{ind}Type: {info['type']}")
    
    if info['type'] == 'Tensor':
        print(f"{ind}Shape: {info['shape']}")
        print(f"{ind}Dtype: {info['dtype']}")
        print(f"{ind}Size: {info['size_mb']:.2f} MB")
        print(f"{ind}Range: [{info['min']}, {info['max']}]")
        
        if 'num_padding' in info:
            print(f"{ind}Padding: {info['num_padding']:,} elements ({info['padding_pct']:.2f}%)")
            print(f"{ind}Valid range: [{info['valid_min']}, {info['valid_max']}]")
            print(f"{ind}Unique values: {info['num_unique_values']:,}")
    
    elif info['type'] == 'dict':
        print(f"{ind}Keys: {info['keys']}")
        for key, details in info['key_details'].items():
            if details['type'] == 'list':
                print(f"{ind}  {key}: list[{details['length']}]")
            elif details['type'] == 'tensor':
                print(f"{ind}  {key}: {details['shape']} {details['dtype']}")


def compare_files(info1, info2):
    """Compare two file infos and report differences"""
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)
    
    # Type comparison
    print("\n1. TYPE COMPARISON")
    print("-" * 100)
    if info1['type'] == info2['type']:
        print(f"✓ Both files have the same type: {info1['type']}")
    else:
        print(f"✗ Different types!")
        print(f"  File 1: {info1['type']}")
        print(f"  File 2: {info2['type']}")
        return  # Can't compare further if types differ
    
    # Tensor comparison
    if info1['type'] == 'Tensor':
        print("\n2. TENSOR SHAPE COMPARISON")
        print("-" * 100)
        
        if info1['shape'] == info2['shape']:
            print(f"✓ Same shape: {info1['shape']}")
        else:
            print(f"✗ Different shapes!")
            print(f"  File 1: {info1['shape']}")
            print(f"  File 2: {info2['shape']}")
        
        print("\n3. DATA TYPE COMPARISON")
        print("-" * 100)
        if info1['dtype'] == info2['dtype']:
            print(f"✓ Same dtype: {info1['dtype']}")
        else:
            print(f"✗ Different dtypes!")
            print(f"  File 1: {info1['dtype']}")
            print(f"  File 2: {info2['dtype']}")
        
        print("\n4. VALUE RANGE COMPARISON")
        print("-" * 100)
        print(f"File 1 range: [{info1['min']}, {info1['max']}]")
        print(f"File 2 range: [{info2['min']}, {info2['max']}]")
        
        if 'num_padding' in info1 and 'num_padding' in info2:
            print(f"\nPadding:")
            print(f"  File 1: {info1['num_padding']:,} ({info1['padding_pct']:.2f}%)")
            print(f"  File 2: {info2['num_padding']:,} ({info2['padding_pct']:.2f}%)")
            
            print(f"\nValid ranges:")
            print(f"  File 1: [{info1['valid_min']}, {info1['valid_max']}]")
            print(f"  File 2: [{info2['valid_min']}, {info2['valid_max']}]")
            
            print(f"\nUnique values:")
            print(f"  File 1: {info1['num_unique_values']:,}")
            print(f"  File 2: {info2['num_unique_values']:,}")
        
        # Element-wise comparison
        print("\n5. ELEMENT-WISE COMPARISON")
        print("-" * 100)
        
        data1 = info1['data']
        data2 = info2['data']
        
        if data1.shape == data2.shape:
            are_equal = torch.equal(data1, data2)
            if are_equal:
                print("✓ Files are IDENTICAL (all elements match)")
            else:
                print("✗ Files are DIFFERENT (elements differ)")
                
                # Calculate differences
                diff_mask = data1 != data2
                num_diff = diff_mask.sum().item()
                pct_diff = (num_diff / data1.numel() * 100)
                
                print(f"\nDifference statistics:")
                print(f"  Different elements: {num_diff:,} / {data1.numel():,} ({pct_diff:.2f}%)")
                
                if len(data1.shape) == 2:
                    # Per-row differences
                    row_diff = diff_mask.any(dim=1)
                    num_diff_rows = row_diff.sum().item()
                    print(f"  Rows with differences: {num_diff_rows:,} / {data1.shape[0]:,}")
                    
                    # Show some examples
                    diff_indices = torch.where(row_diff)[0][:5]
                    if len(diff_indices) > 0:
                        print(f"\nFirst {len(diff_indices)} differing rows:")
                        for idx in diff_indices:
                            idx = idx.item()
                            print(f"\n  Row {idx}:")
                            print(f"    File 1: {data1[idx][:10].tolist()}")
                            print(f"    File 2: {data2[idx][:10].tolist()}")
        else:
            print("✗ Cannot compare element-wise (different shapes)")
    
    # Dict comparison
    elif info1['type'] == 'dict':
        print("\n2. DICTIONARY KEYS COMPARISON")
        print("-" * 100)
        
        keys1 = set(info1['keys'])
        keys2 = set(info2['keys'])
        
        if keys1 == keys2:
            print(f"✓ Same keys: {sorted(keys1)}")
        else:
            print(f"✗ Different keys!")
            only_in_1 = keys1 - keys2
            only_in_2 = keys2 - keys1
            common = keys1 & keys2
            
            if only_in_1:
                print(f"  Only in File 1: {sorted(only_in_1)}")
            if only_in_2:
                print(f"  Only in File 2: {sorted(only_in_2)}")
            print(f"  Common keys: {sorted(common)}")
        
        # Compare common keys
        common_keys = keys1 & keys2
        if common_keys:
            print("\n3. VALUE COMPARISON FOR COMMON KEYS")
            print("-" * 100)
            
            for key in sorted(common_keys):
                det1 = info1['key_details'][key]
                det2 = info2['key_details'][key]
                
                print(f"\nKey '{key}':")
                if det1['type'] == det2['type']:
                    print(f"  Type: {det1['type']}")
                    
                    if det1['type'] == 'list':
                        if det1['length'] == det2['length']:
                            print(f"  ✓ Same length: {det1['length']}")
                            
                            # Compare list contents
                            list1 = info1['data'][key]
                            list2 = info2['data'][key]
                            if list1 == list2:
                                print(f"  ✓ Lists are identical")
                            else:
                                print(f"  ✗ Lists differ")
                                # Count differences
                                diffs = sum(1 for a, b in zip(list1, list2) if a != b)
                                print(f"    Different elements: {diffs} / {len(list1)}")
                        else:
                            print(f"  ✗ Different lengths: {det1['length']} vs {det2['length']}")
                    
                    elif det1['type'] == 'tensor':
                        if det1['shape'] == det2['shape']:
                            print(f"  ✓ Same shape: {det1['shape']}")
                            
                            # Compare tensor
                            t1 = info1['data'][key]
                            t2 = info2['data'][key]
                            if torch.equal(t1, t2):
                                print(f"  ✓ Tensors are identical")
                            else:
                                print(f"  ✗ Tensors differ")
                        else:
                            print(f"  ✗ Different shapes: {det1['shape']} vs {det2['shape']}")
                else:
                    print(f"  ✗ Different types: {det1['type']} vs {det2['type']}")


def main():
    import argparse
    
    # Default paths
    DEFAULT_FILE1 = "explanation/RAWExplainer/NBFNet/FB15k-237/run1/test_explanations_factual_eval_hard_edge_mask_top_k_25.pt"
    DEFAULT_FILE2 = "explanation/RAWExplainer/NBFNet/FB15k-237/run2/test_explanations_factual_eval_hard_edge_mask_top_k_25.pt"
    
    parser = argparse.ArgumentParser(
        description='Compare two explanation .pt files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
                Examples:
                # Use default paths (modify defaults in script as needed)
                python compare_expl_files.py
                
                # Override with custom paths
                python compare_expl_files.py \\
                    --file1 run1/test_explanations_factual_eval_hard_edge_mask_top_k_25.pt \\
                    --file2 run2/test_explanations_factual_eval_hard_edge_mask_top_k_25.pt
                
                # Or use positional arguments
                python compare_expl_files.py file1.pt file2.pt
        """
    )
    
    parser.add_argument(
        'file1',
        nargs='?',
        default=DEFAULT_FILE1,
        help=f'First file to compare (default: {DEFAULT_FILE1})'
    )
    parser.add_argument(
        'file2',
        nargs='?',
        default=DEFAULT_FILE2,
        help=f'Second file to compare (default: {DEFAULT_FILE2})'
    )
    parser.add_argument(
        '--file1',
        dest='file1_named',
        help='First file to compare (named argument)'
    )
    parser.add_argument(
        '--file2',
        dest='file2_named',
        help='Second file to compare (named argument)'
    )
    
    args = parser.parse_args()
    
    # Prioritize named arguments over positional
    file1 = args.file1_named if args.file1_named else args.file1
    file2 = args.file2_named if args.file2_named else args.file2
    
    print("="*100)
    print("EXPLANATION FILE COMPARISON TOOL")
    print("="*100)
    print(f"\nFile 1: {file1}")
    print(f"File 2: {file2}")
    
    # Load and analyze both files
    print("\n" + "="*100)
    print("FILE 1 ANALYSIS")
    print("="*100)
    info1 = load_and_describe(file1)
    if info1:
        print_info(info1)
    
    print("\n" + "="*100)
    print("FILE 2 ANALYSIS")
    print("="*100)
    info2 = load_and_describe(file2)
    if info2:
        print_info(info2)
    
    # Compare
    if info1 and info2:
        compare_files(info1, info2)
    else:
        print("\nERROR: Could not load one or both files. Comparison aborted.")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
