#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_tokenizer_symbols.py
# author: Selvaraj Mani
# date: 10/04/2025
# Purpose: Test SentencePiece tokenizer to validate user-defined symbols integrity and unique token IDs.

import sentencepiece as spm
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

# Import the user_defined_symbols from the trainer
try:
    sys.path.append(str(Path(__file__).parent.parent / "tokenizer"))
    from sptrainer import user_defined_symbols
except ImportError:
    # Fallback: define the symbols directly if import fails
    user_defined_symbols = [
        "```",
        "```cisco-config",
        "```cisco-exec", 
        "```cisco-output",
        "```cisco-syntax",
        "```language",
        "```log",
        "```tcl",
        "<|answer|>",
        "<|doc|>",
        "<|end-answer|>",
        "<|end-question|>",
        "<|end-system|>",
        "<|interface|>",
        "<|ip|>",
        "<|mac|>",
        "<|question|>",
        "<|ssid|>",
        "<|ssid-name|>",
        "<|system|>",
        "<|timestamp|>",
        "<|url|>",
        "<|user|>",
        "<|vlan-id|>",
        "802.1X",
        "QoS",
        "STP",
        "aaa",
        "access-list",
        "access-session",
        "accounting",
        "acl",
        "action",
        "args",
        "authenticate",
        "authentication",
        "authorize",
        "authorization",
        "bay",
        "bgp",
        "binos",
        "brief",
        "bshell",
        "btdecode",
        "btrace",
        "callhome",
        "capture",
        "certificate",
        "chassis",
        "class-map",
        "clear",
        "cisco",
        "cmand",
        "coa",
        "command",
        "configure",
        "cpu",
        "crypto",
        "cts",
        "database",
        "dbm",
        "debug",
        "description",
        "details",
        "dhcp",
        "disable",
        "dns",
        "dot1x",
        "enable",
        "encryption",
        "ethernet",
        "evpn",
        "event",
        "execute",
        "explanation",
        "fman_rp",
        "free",
        "fru",
        "function",
        "gre",
        "hardware",
        "http",
        "icmp",
        "identity",
        "igmp",
        "interface",
        "internal",
        "iosd",
        "ip",
        "ipv6",
        "lacp",
        "license",
        "logging",
        "login",
        "logs",
        "mac",
        "mab",
        "malloc",
        "maroon",
        "memeory",
        "metadata",
        "monitor",
        "mpls",
        "mqipc",
        "node",
        "olaf",
        "password",
        "pcap",
        "platform",
        "polcy-map",
        "policy",
        "port",
        "process",
        "prompt",
        "radius",
        "rbacl",
        "reasoning",
        "redundancy",
        "request",
        "response",
        "routing",
        "server",
        "service",
        "service-policy",
        "shell",
        "shell-manager",
        "show",
        "sicon",
        "sipcap",
        "slot",
        "smart",
        "software",
        "spanning-tree",
        "ssl",
        "standby",
        "status",
        "step_conf",
        "step_db",
        "step_exec",
        "step_explain",
        "step_function",
        "step_internal",
        "step_summary",
        "step_trace",
        "step_type",
        "steps",
        "subscription",
        "summary",
        "switch",
        "sxp",
        "syslog",
        "table",
        "tcp",
        "telemetry",
        "timeout",
        "traffic-shape",
        "troubleshoot",
        "trunk",
        "type",
        "udp",
        "using",
        "vlan",
        "vty",
        "vrf",
        "webauth",
        "wireless",
        "wlan",
    ]

def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    """Load the SentencePiece tokenizer model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp

def test_special_token_ids(sp: spm.SentencePieceProcessor) -> Dict[str, bool]:
    """Test that special tokens have the correct IDs."""
    results = {}
    
    # Expected special token IDs based on sptrainer.py configuration
    expected_ids = {
        'pad': 0,
        'bos': 1, 
        'eos': 2,
        'unk': 3
    }
    
    # Test each special token
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
            
            results[f"{token_name}_id"] = actual_id == expected_id
            
            if actual_id == expected_id:
                logging.info(f"✅ {token_name.upper()} ID correct: {actual_id}")
            else:
                logging.error(f"❌ {token_name.upper()} ID incorrect: expected {expected_id}, got {actual_id}")
                
        except Exception as e:
            logging.error(f"❌ Error getting {token_name} ID: {e}")
            results[f"{token_name}_id"] = False
    
    return results

def test_user_defined_symbols_integrity(sp: spm.SentencePieceProcessor) -> Dict[str, List]:
    """Test that user-defined symbols are not split during tokenization."""
    results = {
        'passed': [],
        'failed': [],
        'missing': []
    }
    
    logging.info(f"Testing {len(user_defined_symbols)} user-defined symbols for integrity...")
    
    for symbol in user_defined_symbols:
        try:
            # Tokenize the symbol
            tokens = sp.encode_as_pieces(symbol)
            
            if len(tokens) == 1 and tokens[0] == symbol:
                results['passed'].append(symbol)
                logging.debug(f"✅ Symbol '{symbol}' tokenized as single token")
            else:
                results['failed'].append((symbol, tokens))
                logging.warning(f"❌ Symbol '{symbol}' split into: {tokens}")
                
        except Exception as e:
            results['missing'].append((symbol, str(e)))
            logging.error(f"❌ Error tokenizing symbol '{symbol}': {e}")
    
    return results

def test_unique_token_ids(sp: spm.SentencePieceProcessor) -> Dict[str, any]:
    """Test that all user-defined symbols have unique token IDs."""
    symbol_to_id = {}
    id_to_symbols = {}
    duplicates = []
    
    logging.info("Testing user-defined symbols for unique token IDs...")
    
    for symbol in user_defined_symbols:
        try:
            token_id = sp.piece_to_id(symbol)
            symbol_to_id[symbol] = token_id
            
            if token_id in id_to_symbols:
                # Duplicate ID found
                id_to_symbols[token_id].append(symbol)
                if len(id_to_symbols[token_id]) == 2:  # First duplicate
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
        'unique_count': len([id for id, symbols in id_to_symbols.items() if len(symbols) == 1])
    }
    
    return results

def test_tokenizer_vocabulary_coverage(sp: spm.SentencePieceProcessor) -> Dict[str, any]:
    """Test vocabulary coverage and stats."""
    vocab_size = sp.get_piece_size()
    
    # Check if all user-defined symbols are in vocabulary
    symbols_in_vocab = 0
    symbols_not_in_vocab = []
    
    for symbol in user_defined_symbols:
        token_id = sp.piece_to_id(symbol)
        if token_id != sp.unk_id():  # Not UNK token
            symbols_in_vocab += 1
        else:
            symbols_not_in_vocab.append(symbol)
    
    return {
        'vocab_size': vocab_size,
        'symbols_in_vocab': symbols_in_vocab,
        'symbols_not_in_vocab': symbols_not_in_vocab,
        'coverage_percentage': (symbols_in_vocab / len(user_defined_symbols)) * 100
    }

def run_comprehensive_test(model_path: Path) -> Dict[str, any]:
    """Run all tests and return comprehensive results."""
    logging.info(f"Loading tokenizer model: {model_path}")
    
    try:
        sp = load_tokenizer(model_path)
        logging.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return {'error': str(e)}
    
    results = {}
    
    # Test 1: Special token IDs
    logging.info("\n" + "="*50)
    logging.info("TEST 1: Special Token IDs")
    logging.info("="*50)
    results['special_tokens'] = test_special_token_ids(sp)
    
    # Test 2: User-defined symbol integrity
    logging.info("\n" + "="*50)
    logging.info("TEST 2: User-Defined Symbol Integrity")
    logging.info("="*50)
    results['symbol_integrity'] = test_user_defined_symbols_integrity(sp)
    
    # Test 3: Unique token IDs
    logging.info("\n" + "="*50)
    logging.info("TEST 3: Unique Token IDs")
    logging.info("="*50)
    results['unique_ids'] = test_unique_token_ids(sp)
    
    # Test 4: Vocabulary coverage
    logging.info("\n" + "="*50)
    logging.info("TEST 4: Vocabulary Coverage")
    logging.info("="*50)
    results['vocabulary_coverage'] = test_tokenizer_vocabulary_coverage(sp)
    
    return results

def print_detailed_results(results: Dict[str, any]) -> None:
    """Print detailed test results."""
    if 'error' in results:
        print(f"\n❌ ERROR: {results['error']}")
        return
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TOKENIZER TEST RESULTS")
    print("="*70)
    
    # Special tokens results
    print("\n🔸 SPECIAL TOKEN IDS:")
    special_results = results['special_tokens']
    all_special_passed = all(special_results.values())
    status = "✅ PASSED" if all_special_passed else "❌ FAILED"
    print(f"   Overall: {status}")
    for token, passed in special_results.items():
        status_icon = "✅" if passed else "❌"
        print(f"   {status_icon} {token}: {passed}")
    
    # Symbol integrity results
    print("\n🔸 SYMBOL INTEGRITY:")
    integrity_results = results['symbol_integrity']
    passed_count = len(integrity_results['passed'])
    failed_count = len(integrity_results['failed'])
    missing_count = len(integrity_results['missing'])
    total_symbols = len(user_defined_symbols)
    
    print(f"   Passed: {passed_count}/{total_symbols} symbols")
    print(f"   Failed: {failed_count}/{total_symbols} symbols")
    print(f"   Missing: {missing_count}/{total_symbols} symbols")
    
    if failed_count > 0:
        print("\n   ❌ FAILED SYMBOLS (split during tokenization):")
        for symbol, tokens in integrity_results['failed']:
            print(f"      '{symbol}' → {tokens}")
    
    if missing_count > 0:
        print("\n   ❌ MISSING SYMBOLS (errors during tokenization):")
        for symbol, error in integrity_results['missing']:
            print(f"      '{symbol}': {error}")
    
    # Unique IDs results
    print("\n🔸 UNIQUE TOKEN IDS:")
    unique_results = results['unique_ids']
    unique_count = unique_results['unique_count']
    duplicate_ids = unique_results['duplicates']
    
    print(f"   Unique symbols: {unique_count}/{total_symbols}")
    print(f"   Duplicate IDs: {len(duplicate_ids)}")
    
    if duplicate_ids:
        print("\n   ❌ DUPLICATE TOKEN IDS:")
        for token_id in duplicate_ids:
            symbols = unique_results['id_to_symbols'][token_id]
            print(f"      ID {token_id}: {symbols}")
    
    # Vocabulary coverage
    print("\n🔸 VOCABULARY COVERAGE:")
    vocab_results = results['vocabulary_coverage']
    print(f"   Total vocabulary size: {vocab_results['vocab_size']}")
    print(f"   Symbols in vocabulary: {vocab_results['symbols_in_vocab']}/{total_symbols}")
    print(f"   Coverage: {vocab_results['coverage_percentage']:.1f}%")
    
    if vocab_results['symbols_not_in_vocab']:
        print("\n   ❌ SYMBOLS NOT IN VOCABULARY:")
        for symbol in vocab_results['symbols_not_in_vocab']:
            print(f"      '{symbol}'")
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY:")
    all_tests_passed = (
        all_special_passed and 
        failed_count == 0 and 
        missing_count == 0 and 
        len(duplicate_ids) == 0 and
        vocab_results['coverage_percentage'] == 100.0
    )
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Tokenizer is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review the issues above.")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Test SentencePiece tokenizer for user-defined symbols integrity and unique token IDs."
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
        help="Suppress detailed logging, show only final results"
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
    
    # Run tests
    results = run_comprehensive_test(model_path)
    
    # Print results
    print_detailed_results(results)

if __name__ == "__main__":
    main()