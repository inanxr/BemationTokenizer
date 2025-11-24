import sentencepiece as spm

def main():
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
    
    print("=== Bemation Tokenizer Examples ===\n")
    
    # Bengali example
    bengali_text = "আমি বাংলাদেশ থেকে এসেছি। আমি একটি ভাষা মডেল তৈরি করছি।"
    bengali_tokens = sp.encode(bengali_text, out_type=int)
    print(f"Bengali text: {bengali_text}")
    print(f"Tokens: {bengali_tokens}")
    print(f"Token count: {len(bengali_tokens)}")
    print(f"Decoded: {sp.decode(bengali_tokens)}\n")
    
    # English example
    english_text = "Once upon a time, there was a little girl who loved to read."
    english_tokens = sp.encode(english_text, out_type=int)
    print(f"English text: {english_text}")
    print(f"Tokens: {english_tokens}")
    print(f"Token count: {len(english_tokens)}")
    print(f"Decoded: {sp.decode(english_tokens)}\n")
    
    # Mixed example
    mixed_text = "I am learning বাংলা language using AI."
    mixed_tokens = sp.encode(mixed_text, out_type=int)
    print(f"Mixed text: {mixed_text}")
    print(f"Tokens: {mixed_tokens}")
    print(f"Token count: {len(mixed_tokens)}")
    print(f"Decoded: {sp.decode(mixed_tokens)}")

if __name__ == "__main__":
    main()
