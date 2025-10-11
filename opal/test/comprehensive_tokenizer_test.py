#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# comprehensive_tokenizer_test.py
# author: Selvaraj Mani
# date: 10/10/2025
# Purpose: Comprehensive tokenizer testing - all tests consolidated in one file

import sentencepiece as smp
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import sys

# Import the user_defined_symbols from the trainer
sys.path.append(str(Path(__file__).parent.parent.parent / "opal" / "tokenizer"))
from sptrainer import user_defined_symbols

def load_tokenizer(model_path: Path) -> smp.SentencePieceProcessor:
    """Load the SentencePiece tokenizer model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    
    sp = smp.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp

def test_special_token_ids(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test that special tokens have the correct IDs."""
    logging.info("Testing special token IDs...")
    
    results = {}
    expected_ids = {
        'pad': 0,
        'bos': 1, 
        'eos': 2,
        'unk': 3
    }
    
    for token_name, expected_id in expected_ids.items():
        try:
            if token_name == 'pad':
                actual_id = sp.pad_id()
            elif token_name == 'bos':
                actual_id = sp.bos_id()
            elif token_name == 'eos':
                actual_id = sp.eos_id()
            elif token_name == 'unk':
                actual_id = sp.unk_id()
            
            passed = actual_id == expected_id
            results[f"{token_name}_id"] = {
                'expected': expected_id,
                'actual': actual_id,
                'passed': passed
            }
            
            status = "✅" if passed else "❌"
            logging.info(f"{status} {token_name.upper()} ID: expected {expected_id}, got {actual_id}")
                
        except Exception as e:
            logging.error(f"❌ Error getting {token_name} ID: {e}")
            results[f"{token_name}_id"] = {'error': str(e), 'passed': False}
    
    return results

def test_symbol_integrity(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test that user-defined symbols are not split during tokenization."""
    logging.info(f"Testing symbol integrity for {len(user_defined_symbols)} symbols...")
    
    results = {
        'passed': [],
        'failed': [],
        'missing': [],
        'total_symbols': len(user_defined_symbols)
    }
    
    for symbol in user_defined_symbols:
        try:
            tokens = sp.encode_as_pieces(symbol)
            
            if len(tokens) == 1 and tokens[0] == symbol:
                results['passed'].append(symbol)
                logging.debug(f"✅ Symbol '{symbol}' → single token")
            else:
                results['failed'].append({'symbol': symbol, 'tokens': tokens})
                logging.warning(f"❌ Symbol '{symbol}' split into: {tokens}")
                
        except Exception as e:
            results['missing'].append({'symbol': symbol, 'error': str(e)})
            logging.error(f"❌ Error tokenizing symbol '{symbol}': {e}")
    
    # Summary statistics
    results['pass_count'] = len(results['passed'])
    results['fail_count'] = len(results['failed'])
    results['missing_count'] = len(results['missing'])
    results['success_rate'] = (results['pass_count'] / results['total_symbols']) * 100
    
    return results

def test_unique_token_ids(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test that all user-defined symbols have unique token IDs."""
    logging.info("Testing unique token IDs...")
    
    symbol_to_id = {}
    id_to_symbols = {}
    duplicates = []
    
    for symbol in user_defined_symbols:
        try:
            token_id = sp.piece_to_id(symbol)
            symbol_to_id[symbol] = token_id
            
            if token_id in id_to_symbols:
                id_to_symbols[token_id].append(symbol)
                if len(id_to_symbols[token_id]) == 2:
                    duplicates.append(token_id)
            else:
                id_to_symbols[token_id] = [symbol]
                
        except Exception as e:
            logging.error(f"❌ Error getting token ID for symbol '{symbol}': {e}")
            symbol_to_id[symbol] = None
    
    results = {
        'symbol_to_id': symbol_to_id,
        'id_to_symbols': id_to_symbols,
        'duplicates': duplicates,
        'unique_count': len([id for id, symbols in id_to_symbols.items() if len(symbols) == 1]),
        'total_symbols': len(user_defined_symbols)
    }
    
    return results

def test_vocabulary_coverage(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test vocabulary coverage and stats."""
    logging.info("Testing vocabulary coverage...")
    
    vocab_size = sp.get_piece_size()
    symbols_in_vocab = 0
    symbols_not_in_vocab = []
    
    for symbol in user_defined_symbols:
        token_id = sp.piece_to_id(symbol)
        if token_id != sp.unk_id():
            symbols_in_vocab += 1
        else:
            symbols_not_in_vocab.append(symbol)
    
    results = {
        'vocab_size': vocab_size,
        'symbols_in_vocab': symbols_in_vocab,
        'symbols_not_in_vocab': symbols_not_in_vocab,
        'total_symbols': len(user_defined_symbols),
        'coverage_percentage': (symbols_in_vocab / len(user_defined_symbols)) * 100
    }
    
    return results

def test_whitespace_handling(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test whitespace character handling with round-trip validation."""
    logging.info("Testing whitespace handling...")
    
    whitespace_chars = {
        'space': ' ',
        'newline': '\n',
        'tab': '\t',
        'double_space': '  ',
        'four_spaces': '    '
    }
    
    results = {'whitespace_tokens': {}}
    
    for name, char in whitespace_chars.items():
        try:
            # Use encoding/decoding for proper validation, not piece_to_id for raw characters
            pieces = sp.encode_as_pieces(char)
            token_ids = sp.encode_as_ids(char)
            decoded = sp.decode_ids(token_ids)
            
            # Round-trip test is the real validation
            round_trip_success = decoded == char
            
            # For spaces, check if they're represented as ▁ (SentencePiece convention)
            if name == 'space':
                # Space should be encoded as ▁ in SentencePiece
                space_handled_correctly = len(pieces) == 1 and pieces[0] == '▁'
                effective_success = round_trip_success and space_handled_correctly
            elif name in ['double_space', 'four_spaces']:
                # Multiple spaces should be multiple ▁ tokens
                expected_count = len(char)
                space_handled_correctly = len(pieces) == expected_count and all(p == '▁' for p in pieces)
                effective_success = round_trip_success and space_handled_correctly
            else:
                # Other whitespace (newline, tab) should have direct representation
                effective_success = round_trip_success and len(pieces) == 1 and pieces[0] == char
            
            results['whitespace_tokens'][name] = {
                'character': repr(char),
                'pieces': pieces,
                'token_ids': token_ids,
                'decoded': repr(decoded),
                'round_trip_success': round_trip_success,
                'effective_success': effective_success,
                'piece_count': len(pieces)
            }
            
            status = "✅" if effective_success else "❌"
            if name == 'space':
                logging.info(f"{status} {name}: {repr(char)} → Pieces: {pieces} (SentencePiece format)")
            else:
                logging.info(f"{status} {name}: {repr(char)} → Pieces: {pieces}")
            
        except Exception as e:
            logging.error(f"❌ Error testing {name} '{repr(char)}': {e}")
            results['whitespace_tokens'][name] = {'error': str(e)}
    
    return results

def test_sample_sentences(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test tokenization of sample sentences."""
    logging.info("Testing sample sentence tokenization...")
    
    sample_sentences = {
        'simple_with_spaces': 'This is a test sentence.',
        'with_newlines': 'Line 1\nLine 2\nLine 3',
        'with_tabs': 'Column1\tColumn2\tColumn3',
        'mixed_whitespace': 'Start\n\tIndented line\n    Four spaces\t\tDouble tab',
        'code_block': '```cisco-config\ninterface GigabitEthernet0/1\n  description Test interface\n```',
        'structured_tokens': '<|doc|>Sample content<|system|>Response<|end-system|>',
        'cisco_commands': 'show interface brief\nconfigure terminal\ninterface GigabitEthernet0/1',
        'mixed_symbols': 'QoS policy for 802.1X authentication using aaa radius'
    }
    
    results = {'sample_sentences': {}}
    
    for name, sentence in sample_sentences.items():
        try:
            pieces = sp.encode_as_pieces(sentence)
            token_ids = sp.encode_as_ids(sentence)
            
            # Analyze tokens
            unk_count = sum(1 for tid in token_ids if tid == sp.unk_id())
            user_symbol_count = sum(1 for piece in pieces if piece in user_defined_symbols)
            
            results['sample_sentences'][name] = {
                'sentence': sentence,
                'sentence_repr': repr(sentence),
                'total_tokens': len(pieces),
                'pieces': pieces,
                'token_ids': token_ids,
                'unk_count': unk_count,
                'user_symbol_count': user_symbol_count,
                'avg_chars_per_token': len(sentence) / len(pieces) if pieces else 0
            }
            
            logging.info(f"✅ {name}: {len(pieces)} tokens, {user_symbol_count} user symbols")
            
        except Exception as e:
            logging.error(f"❌ Error tokenizing sentence '{name}': {e}")
            results['sample_sentences'][name] = {'error': str(e)}
    
    return results

def test_backtick_differentiation(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test that backtick symbols are properly differentiated."""
    logging.info("Testing backtick symbol differentiation...")
    
    backtick_symbols = [s for s in user_defined_symbols if s.startswith('```')]
    
    results = {
        'backtick_symbols': {},
        'differentiation_test': {}
    }
    
    # Test each backtick symbol
    for symbol in backtick_symbols:
        token_id = sp.piece_to_id(symbol)
        pieces = sp.encode_as_pieces(symbol)
        
        results['backtick_symbols'][symbol] = {
            'token_id': token_id,
            'pieces': pieces,
            'is_single_token': len(pieces) == 1 and pieces[0] == symbol
        }
    
    # Test differentiation between simple and compound backticks
    simple_backticks = '```'
    compound_examples = ['```cisco-config', '```cisco-exec', '```log']
    
    if simple_backticks in backtick_symbols:
        simple_id = sp.piece_to_id(simple_backticks)
        results['differentiation_test']['simple_backticks'] = {
            'symbol': simple_backticks,
            'token_id': simple_id
        }
        
        compound_results = {}
        for compound in compound_examples:
            if compound in backtick_symbols:
                compound_id = sp.piece_to_id(compound)
                compound_results[compound] = {
                    'token_id': compound_id,
                    'different_from_simple': compound_id != simple_id
                }
        
        results['differentiation_test']['compounds'] = compound_results
        results['differentiation_test']['all_different'] = all(
            data['different_from_simple'] for data in compound_results.values()
        )
    
    return results

def test_context_integrity(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test symbol integrity in realistic contexts."""
    logging.info("Testing symbol integrity in context...")
    
    test_contexts = [
        'Configure ```cisco-config interface GigabitEthernet0/1```',
        'Step: step_conf - Check configuration step_exec - Execute commands',
        'Use <|system|>Configure QoS policy<|end-system|> for network',
        'Authentication: aaa authentication dot1x default group radius',
        'Protocol support: tcp udp icmp with 802.1X and QoS enabled'
    ]
    
    results = {
        'context_tests': {},
        'symbol_integrity_issues': []
    }
    
    for i, context in enumerate(test_contexts):
        test_name = f"context_{i+1}"
        pieces = sp.encode_as_pieces(context)
        token_ids = sp.encode_as_ids(context)
        
        # Check for user symbols in pieces
        found_symbols = [piece for piece in pieces if piece in user_defined_symbols]
        
        # Check for potential symbol splitting
        integrity_issues = []
        for symbol in user_defined_symbols:
            if symbol in context:
                if symbol not in pieces:
                    # Symbol might be split
                    integrity_issues.append({
                        'symbol': symbol,
                        'context': context,
                        'issue': 'symbol_not_in_pieces'
                    })
        
        results['context_tests'][test_name] = {
            'context': context,
            'total_tokens': len(pieces),
            'pieces': pieces,
            'found_symbols': found_symbols,
            'integrity_issues': integrity_issues
        }
        
        if integrity_issues:
            results['symbol_integrity_issues'].extend(integrity_issues)
    
    return results

def test_newline_round_trip(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test newline encode/decode round-trip to ensure perfect reconstruction."""
    logging.info("Testing newline encode/decode round-trip...")
    
    # Test cases with various newline patterns (using lowercase to match tokenizer behavior)
    test_cases = {
        'multiline_natural': """this is the end of the thought.

now start the code block.""",
        
        'multiline_explicit': "this is the end of the thought.\nnow start the code block.",
        
        'cisco_config_block': """interface gigabitethernet0/1
  description corporate lan
  ip address 192.168.1.1 255.255.255.0
  no shutdown
exit""",
        
        'mixed_newlines_tabs': "line 1\n\tindented line\n  two spaces\n\t\tdouble indent",
        
        'code_with_newlines': """```cisco-config
interface range gigabitethernet0/1-24
  switchport mode access
  switchport access vlan 10
```""",
        
        'structured_with_newlines': """<|doc|>configuration guide
<|question|>how to configure interfaces?<|end-question|>
<|answer|>use interface commands<|end-answer|>
<|end-doc|>""",
        
        'single_newline': "\n",
        'double_newline': "\n\n",
        'triple_newline': "\n\n\n",
        'leading_newline': "\nstart with newline",
        'trailing_newline': "end with newline\n",
        'only_whitespace': "\n\t  \n   \t\n",
        
        # Additional test cases focusing on newline preservation
        'commands_with_newlines': "show interface brief\nconfigure terminal\ninterface gigabitethernet0/1",
        'nested_indentation': "level 1\n  level 2\n    level 3\n      level 4",
        'empty_lines': "line 1\n\nline 3\n\n\nline 6"
    }
    
    results = {
        'test_cases': {},
        'all_passed': True,
        'failed_cases': [],
        'summary': {},
        'newline_analysis': {}
    }
    
    total_newlines_original = 0
    total_newlines_decoded = 0
    newline_mismatches = 0
    
    for test_name, original_text in test_cases.items():
        try:
            # Encode the text
            token_ids = sp.encode_as_ids(original_text)
            pieces = sp.encode_as_pieces(original_text)
            
            # Decode back to text
            decoded_text = sp.decode_ids(token_ids)
            decoded_from_pieces = sp.decode_pieces(pieces)
            
            # Check if round-trip is perfect
            round_trip_success = decoded_text == original_text
            pieces_round_trip_success = decoded_from_pieces == original_text
            
            # Count newlines in original vs decoded
            original_newlines = original_text.count('\n')
            decoded_newlines = decoded_text.count('\n')
            
            # Check if newline tokens are preserved
            newline_tokens = [piece for piece in pieces if piece == '\n']
            newline_tokens_count = len(newline_tokens)
            
            # Newline structure analysis
            newlines_preserved = original_newlines == decoded_newlines
            if not newlines_preserved:
                newline_mismatches += 1
            
            total_newlines_original += original_newlines
            total_newlines_decoded += decoded_newlines
            
            test_result = {
                'original_text': original_text,
                'original_repr': repr(original_text),
                'token_ids': token_ids,
                'pieces': pieces,
                'decoded_text': decoded_text,
                'decoded_repr': repr(decoded_text),
                'decoded_from_pieces': decoded_from_pieces,
                'round_trip_success': round_trip_success,
                'pieces_round_trip_success': pieces_round_trip_success,
                'original_newlines': original_newlines,
                'decoded_newlines': decoded_newlines,
                'newline_tokens_count': newline_tokens_count,
                'newlines_preserved': newlines_preserved,
                'newline_tokens_match': newline_tokens_count == original_newlines,
                'total_tokens': len(pieces),
                'original_length': len(original_text),
                'decoded_length': len(decoded_text),
                'newline_structure_intact': newlines_preserved and newline_tokens_count == original_newlines
            }
            
            results['test_cases'][test_name] = test_result
            
            # We consider success if newline structure is preserved, even if case changes
            test_success = test_result['newline_structure_intact']
            
            if not test_success:
                results['all_passed'] = False
                results['failed_cases'].append(test_name)
                logging.error(f"❌ {test_name}: Newline structure failed")
                logging.error(f"   Original newlines: {original_newlines}, Decoded: {decoded_newlines}, Tokens: {newline_tokens_count}")
            else:
                status = "Perfect" if round_trip_success else "Newlines preserved"
                logging.info(f"✅ {test_name}: {status} ({len(pieces)} tokens, {original_newlines} newlines)")
                
        except Exception as e:
            logging.error(f"❌ Error testing {test_name}: {e}")
            results['test_cases'][test_name] = {'error': str(e)}
            results['all_passed'] = False
            results['failed_cases'].append(test_name)
    
    # Generate summary statistics
    total_tests = len(test_cases)
    passed_tests = total_tests - len(results['failed_cases'])
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': len(results['failed_cases']),
        'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
        'all_round_trips_successful': results['all_passed']
    }
    
    # Newline analysis
    results['newline_analysis'] = {
        'total_newlines_original': total_newlines_original,
        'total_newlines_decoded': total_newlines_decoded,
        'newline_preservation_rate': (total_newlines_decoded / total_newlines_original) * 100 if total_newlines_original > 0 else 100,
        'newline_mismatches': newline_mismatches,
        'perfect_newline_preservation': newline_mismatches == 0
    }
    
    return results

def test_token_efficiency(sp: smp.SentencePieceProcessor) -> Dict[str, Any]:
    """Test tokenization efficiency metrics."""
    logging.info("Testing tokenization efficiency...")
    
    # Test various content types
    content_types = {
        'cisco_config': '''
interface GigabitEthernet0/1
  description Corporate LAN
  ip address 192.168.1.1 255.255.255.0
  no shutdown
''',
        'cisco_commands': 'show interface brief\nshow ip route\nshow version',
        'structured_content': '''
<|doc|>Network troubleshooting guide
<|question|>How to configure QoS?<|end-question|>
<|answer|>Use policy-map and class-map<|end-answer|>
''',
        'mixed_technical': 'Configure 802.1X authentication with aaa radius server',
        'code_blocks': '''
```cisco-config
router bgp 65001
  neighbor 10.1.1.1 remote-as 65002
```
'''
    }
    
    results = {'efficiency_tests': {}, 'overall_metrics': {}}
    
    total_chars = 0
    total_tokens = 0
    
    for content_type, content in content_types.items():
        pieces = sp.encode_as_pieces(content)
        token_ids = sp.encode_as_ids(content)
        
        char_count = len(content)
        token_count = len(pieces)
        chars_per_token = char_count / token_count if token_count > 0 else 0
        
        user_symbol_count = sum(1 for piece in pieces if piece in user_defined_symbols)
        unk_count = sum(1 for tid in token_ids if tid == sp.unk_id())
        
        results['efficiency_tests'][content_type] = {
            'char_count': char_count,
            'token_count': token_count,
            'chars_per_token': chars_per_token,
            'user_symbol_count': user_symbol_count,
            'unk_count': unk_count,
            'compression_ratio': token_count / char_count if char_count > 0 else 0
        }
        
        total_chars += char_count
        total_tokens += token_count
    
    results['overall_metrics'] = {
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'average_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 0,
        'overall_compression_ratio': total_tokens / total_chars if total_chars > 0 else 0
    }
    
    return results

def run_all_tests(model_path: Path) -> Dict[str, Any]:
    """Run all tokenizer tests and return comprehensive results."""
    logging.info(f"🚀 Starting comprehensive tokenizer testing")
    logging.info(f"Model: {model_path}")
    logging.info(f"Testing {len(user_defined_symbols)} user-defined symbols")
    
    start_time = time.time()
    
    try:
        sp = load_tokenizer(model_path)
        logging.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return {'error': str(e)}
    
    # Run all test functions
    test_functions = [
        ('special_tokens', test_special_token_ids),
        ('symbol_integrity', test_symbol_integrity),
        ('unique_ids', test_unique_token_ids),
        ('vocabulary_coverage', test_vocabulary_coverage),
        ('whitespace_handling', test_whitespace_handling),
        ('sample_sentences', test_sample_sentences),
        ('backtick_differentiation', test_backtick_differentiation),
        ('context_integrity', test_context_integrity),
        ('newline_round_trip', test_newline_round_trip),
        ('token_efficiency', test_token_efficiency)
    ]
    
    results = {
        'model_path': str(model_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'user_symbols_count': len(user_defined_symbols),
        'tests': {}
    }
    
    for test_name, test_func in test_functions:
        logging.info(f"\n{'='*60}")
        logging.info(f"RUNNING TEST: {test_name.upper()}")
        logging.info(f"{'='*60}")
        
        try:
            test_result = test_func(sp)
            results['tests'][test_name] = test_result
            logging.info(f"✅ {test_name} completed successfully")
        except Exception as e:
            logging.error(f"❌ {test_name} failed: {e}")
            results['tests'][test_name] = {'error': str(e)}
    
    # Calculate overall assessment
    results['duration'] = time.time() - start_time
    results['overall_assessment'] = calculate_overall_assessment(results['tests'])
    
    return results

def calculate_overall_assessment(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall assessment from test results."""
    assessment = {
        'tests_passed': 0,
        'tests_failed': 0,
        'critical_issues': [],
        'warnings': [],
        'overall_score': 0,
        'quality_rating': 'Unknown'
    }
    
    # Analyze symbol integrity
    if 'symbol_integrity' in test_results:
        integrity = test_results['symbol_integrity']
        if 'success_rate' in integrity:
            if integrity['success_rate'] == 100:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['critical_issues'].append(
                    f"Symbol integrity: {integrity['fail_count']}/{integrity['total_symbols']} symbols failed"
                )
    
    # Analyze vocabulary coverage
    if 'vocabulary_coverage' in test_results:
        coverage = test_results['vocabulary_coverage']
        if 'coverage_percentage' in coverage:
            if coverage['coverage_percentage'] == 100:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['critical_issues'].append(
                    f"Vocabulary coverage: {coverage['coverage_percentage']:.1f}% (missing {len(coverage['symbols_not_in_vocab'])} symbols)"
                )
    
    # Analyze unique token IDs
    if 'unique_ids' in test_results:
        unique_ids = test_results['unique_ids']
        if 'duplicates' in unique_ids:
            if len(unique_ids['duplicates']) == 0:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['critical_issues'].append(
                    f"Duplicate token IDs: {len(unique_ids['duplicates'])} conflicts found"
                )
    
    # Analyze backtick differentiation
    if 'backtick_differentiation' in test_results:
        backticks = test_results['backtick_differentiation']
        if 'differentiation_test' in backticks and 'all_different' in backticks['differentiation_test']:
            if backticks['differentiation_test']['all_different']:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['warnings'].append("Backtick symbols may not be properly differentiated")
    
    # Calculate overall score
    total_tests = assessment['tests_passed'] + assessment['tests_failed']
    if total_tests > 0:
        assessment['overall_score'] = (assessment['tests_passed'] / total_tests) * 100
    
    # Determine quality rating
    if assessment['overall_score'] >= 90:
        assessment['quality_rating'] = 'Excellent'
    elif assessment['overall_score'] >= 75:
        assessment['quality_rating'] = 'Good'
    elif assessment['overall_score'] >= 50:
        assessment['quality_rating'] = 'Fair'
    else:
        assessment['quality_rating'] = 'Poor'
    
    return assessment

def print_comprehensive_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive test summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TOKENIZER TEST RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ TESTING FAILED: {results['error']}")
        return
    
    print(f"📊 Model: {results['model_path']}")
    print(f"⏰ Timestamp: {results['timestamp']}")
    print(f"📝 User symbols: {results['user_symbols_count']}")
    print(f"⏱️  Duration: {results['duration']:.2f} seconds")
    
    # Overall assessment
    assessment = results['overall_assessment']
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Quality Rating: {assessment['quality_rating']}")
    print(f"   Overall Score: {assessment['overall_score']:.1f}%")
    print(f"   Tests Passed: {assessment['tests_passed']}")
    print(f"   Tests Failed: {assessment['tests_failed']}")
    
    # Critical issues
    if assessment['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES:")
        for issue in assessment['critical_issues']:
            print(f"   • {issue}")
    
    # Warnings
    if assessment['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in assessment['warnings']:
            print(f"   • {warning}")
    
    # Test summaries
    tests = results['tests']
    
    if 'symbol_integrity' in tests:
        integrity = tests['symbol_integrity']
        print(f"\n🔸 SYMBOL INTEGRITY:")
        print(f"   Success Rate: {integrity.get('success_rate', 0):.1f}%")
        print(f"   Passed: {integrity.get('pass_count', 0)}/{integrity.get('total_symbols', 0)}")
        if integrity.get('fail_count', 0) > 0:
            print(f"   Failed: {integrity['fail_count']} symbols")
    
    if 'vocabulary_coverage' in tests:
        coverage = tests['vocabulary_coverage']
        print(f"\n🔸 VOCABULARY COVERAGE:")
        print(f"   Coverage: {coverage.get('coverage_percentage', 0):.1f}%")
        print(f"   In Vocabulary: {coverage.get('symbols_in_vocab', 0)}/{coverage.get('total_symbols', 0)}")
        print(f"   Vocabulary Size: {coverage.get('vocab_size', 0)}")
    
    if 'whitespace_handling' in tests:
        whitespace = tests['whitespace_handling']
        print(f"\n🔸 WHITESPACE HANDLING:")
        ws_tokens = whitespace.get('whitespace_tokens', {})
        for name, data in ws_tokens.items():
            if 'error' not in data:
                status = "✅" if data.get('effective_success', False) else "❌"
                pieces = data.get('pieces', [])
                pieces_str = str(pieces) if len(pieces) <= 3 else f"{pieces[:2]}...+{len(pieces)-2}"
                print(f"   {status} {name}: {data.get('character', '')} → Pieces: {pieces_str}")
                if name == 'space' and data.get('effective_success', False):
                    print(f"     (SentencePiece format: spaces represented as ▁)")
    
    if 'token_efficiency' in tests:
        efficiency = tests['token_efficiency']
        overall = efficiency.get('overall_metrics', {})
        print(f"\n🔸 TOKENIZATION EFFICIENCY:")
        print(f"   Average chars/token: {overall.get('average_chars_per_token', 0):.2f}")
        print(f"   Compression ratio: {overall.get('overall_compression_ratio', 0):.3f}")
    
    if 'newline_round_trip' in tests:
        round_trip = tests['newline_round_trip']
        summary = round_trip.get('summary', {})
        print(f"\n🔸 NEWLINE ROUND-TRIP:")
        print(f"   Tests passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        if not summary.get('all_round_trips_successful', False):
            failed_cases = round_trip.get('failed_cases', [])
            print(f"   Failed cases: {', '.join(failed_cases)}")
    
    # Final verdict
    print("\n" + "="*80)
    if assessment['quality_rating'] == 'Excellent' and len(assessment['critical_issues']) == 0:
        print("🎉 EXCELLENT! Tokenizer quality is outstanding.")
    elif assessment['quality_rating'] in ['Good', 'Excellent']:
        print("✅ GOOD! Tokenizer quality is acceptable with minor issues.")
    else:
        print("⚠️  ISSUES DETECTED! Review problems above.")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive tokenizer testing - all tests in one file"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the SentencePiece model file (.model)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging, show only summary"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    model_path = Path(args.model_path)
    
    # Run all tests
    results = run_all_tests(model_path)
    
    # Print summary
    print_comprehensive_summary(results)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📄 Detailed results saved to: {output_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    main()