#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify the preprocessing fix works correctly
Run this BEFORE full training to catch issues early
"""

import json
import sys

def test_program_format():
    """Test that program formatting matches evaluation expectations"""
    print("="*60)
    print("TEST 1: Program Format Conversion")
    print("="*60)
    
    # Test case from FinQA
    test_tokens = ['divide', '(', '100', ',', '100', ')']
    
    # What evaluate.py expects as input
    expected_string = "divide( 100, 100)"
    
    # Simulate our conversion
    from preprocess_fixed import program_tokens_to_string, safe_parse_program_field
    
    result = program_tokens_to_string(test_tokens)
    print(f"Input tokens: {test_tokens}")
    print(f"Output string: {result}")
    print(f"Expected format: {expected_string}, EOF")
    
    # Test round-trip
    print("\nTest round-trip (what evaluate.py does):")
    # Simulate evaluate.py's program_tokenization
    parsed = result.split(', ')
    print(f"After split on ', ': {parsed}")
    
    success = 'divide(' in result and 'EOF' in result
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_evaluation_format():
    """Test that our predictions match evaluate.py's expected format"""
    print("\n" + "="*60)
    print("TEST 2: Evaluation Format Compatibility")
    print("="*60)
    
    # Create a test prediction file
    test_prediction = [
        {
            "id": "test_1",
            "predicted": ['divide', '(', '100', ',', '100', ')', 'EOF']
        }
    ]
    
    # Save it
    with open('test_pred.json', 'w') as f:
        json.dump(test_prediction, f)
    
    print("Created test prediction file: test_pred.json")
    print(f"Format: {test_prediction}")
    
    # Check structure
    is_list = isinstance(test_prediction[0]['predicted'], list)
    has_id = 'id' in test_prediction[0]
    has_predicted = 'predicted' in test_prediction[0]
    has_eof = test_prediction[0]['predicted'][-1] == 'EOF'
    
    print(f"\nChecks:")
    print(f"  - Is list: {is_list} {'✅' if is_list else '❌'}")
    print(f"  - Has 'id': {has_id} {'✅' if has_id else '❌'}")
    print(f"  - Has 'predicted': {has_predicted} {'✅' if has_predicted else '❌'}")
    print(f"  - Has EOF: {has_eof} {'✅' if has_eof else '❌'}")
    
    success = all([is_list, has_id, has_predicted, has_eof])
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_tokenization_quality():
    """Test that tokenization preserves program structure"""
    print("\n" + "="*60)
    print("TEST 3: Tokenization Quality")
    print("="*60)
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test string
    test_program = "divide( 100, 100), EOF"
    
    # Tokenize
    tokens = tokenizer.encode(test_program, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {test_program}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Check if key elements are preserved
    has_divide = 'divide' in decoded
    has_numbers = '100' in decoded
    has_eof = 'EOF' in decoded
    
    print(f"\nPreservation checks:")
    print(f"  - Contains 'divide': {has_divide} {'✅' if has_divide else '❌'}")
    print(f"  - Contains '100': {has_numbers} {'✅' if has_numbers else '❌'}")
    print(f"  - Contains 'EOF': {has_eof} {'✅' if has_eof else '❌'}")
    
    success = all([has_divide, has_numbers, has_eof])
    print(f"\n{'✅ PASS' if success else '❌'}")
    return success

def test_with_real_data():
    """Test with actual FinQA data if available"""
    print("\n" + "="*60)
    print("TEST 4: Real Data Compatibility")
    print("="*60)
    
    try:
        # Try to load actual test data
        with open('test.json', 'r') as f:
            data = json.load(f)
        
        print(f"✅ Found test.json with {len(data)} examples")
        
        # Check first example structure
        example = data[0]
        has_qa = 'qa' in example
        has_program = 'program' in example.get('qa', {})
        has_table = 'table_ori' in example
        
        print(f"\nData structure checks:")
        print(f"  - Has 'qa': {has_qa} {'✅' if has_qa else '❌'}")
        print(f"  - Has 'program': {has_program} {'✅' if has_program else '❌'}")
        print(f"  - Has 'table_ori': {has_table} {'✅' if has_table else '❌'}")
        
        if has_program:
            program = example['qa']['program']
            print(f"\nSample program: {program[:50] if len(str(program)) > 50 else program}")
        
        success = all([has_qa, has_program, has_table])
        print(f"\n{'✅ PASS' if success else '❌ FAIL'}")
        return success
        
    except FileNotFoundError:
        print("⚠️  test.json not found - download FinQA data first")
        print("Run: wget https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json")
        return None

def main():
    print("\n" + "🧪 RUNNING PRE-TRAINING TESTS" + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Program Format", test_program_format()))
    results.append(("Evaluation Format", test_evaluation_format()))
    results.append(("Tokenization", test_tokenization_quality()))
    results.append(("Real Data", test_with_real_data()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        if result is None:
            status = "⚠️  SKIPPED"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    # Overall result
    passed = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    print("\n" + "="*60)
    if passed == total and total > 0:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to proceed with training")
    else:
        print(f"⚠️  {total - passed}/{total} tests failed")
        print("❌ Fix issues before training")
    print("="*60)

if __name__ == "__main__":
    main()
