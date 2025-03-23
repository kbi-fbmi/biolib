"""Faundantion model helpers functions"""

import concurrent.futures


def tokenize_sequence_parallel(seq, tokenize_function, max_workers=16):
    print(f"Starting tokenization with {max_workers} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        tokens = list(executor.map(tokenize_function, seq))
    print("Tokenization completed")
    return tokens


def extr_key(dict_list, key):
    print(f"Extracting key '{key}' from dictionary list")
    return [d[key] for d in dict_list]
    return [d[key] for d in dict_list]
