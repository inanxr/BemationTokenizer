import sentencepiece as spm

def main():
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
    
    print("=== Bemation Tokenizer Example ===\n")
    
    # Example 1
    text1 = "This is a sample text for tokenization."
    tokens1 = sp.encode(text1, out_type=int)
    print(f"Text: {text1}")
    print(f"Tokens: {tokens1}")
    print(f"Token count: {len(tokens1)}")
    print(f"Decoded: {sp.decode(tokens1)}\n")
    
    # Example 2
    text2 = "Natural language processing with multilingual support."
    tokens2 = sp.encode(text2, out_type=int)
    print(f"Text: {text2}")
    print(f"Tokens: {tokens2}")
    print(f"Token count: {len(tokens2)}")
    print(f"Decoded: {sp.decode(tokens2)}\n")
    
    # Show vocabulary info
    print(f"Vocabulary size: {sp.vocab_size()}")
    print(f"Piece size: {sp.piece_size()}")

if __name__ == "__main__":
    main()
