"""
Benchmark your tokenizer's efficiency.
"""

import sentencepiece as spm

def benchmark():
    print("="*60)
    print("TOKENIZER BENCHMARK")
    print("="*60)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
    
    print(f"\nVocabulary size: {sp.vocab_size()}")
    print(f"Model type: SentencePiece")
    
    # Test with sample texts
    test_texts = [
        "This is a simple sentence to test the tokenizer.",
        "Natural language processing enables machines to understand human language.",
        "Machine learning models require efficient tokenization for better performance.",
    ]
    
    print(f"\n📊 Performance Metrics:\n")
    
    total_words = 0
    total_tokens = 0
    
    for i, text in enumerate(test_texts, 1):
        tokens = sp.encode(text, out_type=int)
        words = len(text.split())
        total_words += words
        total_tokens += len(tokens)
        
        print(f"Test {i}:")
        print(f"  Text: {text}")
        print(f"  Words: {words}, Tokens: {len(tokens)}")
        print(f"  Tokens/word: {len(tokens)/words:.2f}\n")
    
    avg_ratio = total_tokens / total_words
    print(f"Average tokens/word: {avg_ratio:.2f}")
    
    print("="*60)
    print(f"\n✅ Benchmark complete!")
    print(f"   Lower tokens/word = more efficient tokenization")
    print("="*60)

if __name__ == "__main__":
    benchmark()
