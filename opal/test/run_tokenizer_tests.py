#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_tokenizer_tests.py
# author: Selvaraj Mani
# date: 10/04/2025
# Purpose: Simple test runner for SentencePiece tokenizer validation

"""
Quick test runner for tokenizer validation.

This script provides an easy way to test SentencePiece tokenizers
without needing to remember command line arguments.

Usage:
    python3 run_tokenizer_tests.py
"""

import sys
from pathlib import Path

def main():
    # Add the test directory to Python path
    test_dir = Path(__file__).parent / "opal" / "test"
    sys.path.append(str(test_dir))
    
    try:
        from test_tokenizer_symbols import run_comprehensive_test, print_detailed_results
    except ImportError as e:
        print(f"❌ Error importing test module: {e}")
        print("Make sure you're running this from the project root directory.")
        return 1
    
    # Look for available tokenizer models
    model_paths = [
        Path("checkpoints/pretrained/opal_tokenizer.model"),
        Path("opal_tokenizer.model"),
        Path("tokenizer.model"),
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ No tokenizer model found.")
        print("Searched for:")
        for path in model_paths:
            print(f"  - {path}")
        print("\nPlease train a tokenizer first using:")
        print("  python3 opal/tokenizer/sptrainer.py training_data.txt")
        return 1
    
    print(f"🔍 Found tokenizer model: {model_path}")
    print("🧪 Running comprehensive validation tests...\n")
    
    # Run tests
    results = run_comprehensive_test(model_path)
    
    # Print results
    print_detailed_results(results)
    
    # Return appropriate exit code
    if 'error' in results:
        return 1
    
    # Check if all tests passed
    special_passed = all(results['special_tokens'].values())
    integrity_passed = len(results['symbol_integrity']['failed']) == 0
    unique_passed = len(results['unique_ids']['duplicates']) == 0
    coverage_passed = results['vocabulary_coverage']['coverage_percentage'] == 100.0
    
    if special_passed and integrity_passed and unique_passed and coverage_passed:
        print("\n🎉 All tests passed! Ready for production use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Tokenizer needs attention.")
        return 1

if __name__ == "__main__":
    exit(main())