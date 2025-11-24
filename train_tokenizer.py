"""
Quick tokenizer training script using sample texts.
This trains the Bemation tokenizer without downloading full books.
"""

import sentencepiece as spm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample English texts (public domain)
ENGLISH_SAMPLES = """
Once upon a time, in a faraway land, there lived a young princess who loved to read books.
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
In the beginning was the Word, and the Word was with God, and the Word was God.
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.
Call me Ishmael. Some years ago - never mind how long precisely - having little or no money in my purse.
All happy families are alike; each unhappy family is unhappy in its own way.
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms.
The story so far: In the beginning the Universe was created. This has made a lot of people very angry.
I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe.
Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else.
Happy families are all alike; every unhappy family is unhappy in its own way.
It was a bright cold day in April, and the clocks were striking thirteen.
Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy.
The past is a foreign country; they do things differently there.
""" * 100  # Repeat to increase corpus size

# Sample Bengali texts (public domain, Rabindranath Tagore and others)
BENGALI_SAMPLES = """
আমি বাংলায় গান গাই। আমি বাংলার গান গাই। আমি আমার আমিকে চিরদিন এই বাংলায় খুঁজে পাই।
আমার সোনার বাংলা, আমি তোমায় ভালোবাসি। চিরদিন তোমার আকাশ, তোমার বাতাস, আমার প্রাণে বাজায় বাঁশি।
পথের দেবতা, প্রাণের দেবতা, সত্যের দেবতা। আমার মাথার উপর আকাশে তোমার পবিত্র পাদপীঠ।
জীবন যখন শুকায়ে যায় করুণাধারায় এসো, সকরুণ প্রভু, এসো। হৃদয় যখন থাকে না আর সত্যের মাঝে।
আমার এই পথ চাওয়াতেই আনন্দ। আমার এই কথা বলাতেই আনন্দ। আমার এই গান গাওয়াতেই আনন্দ।
কবির লেখা কবিতা কবিতায় পরিণত হয় যখন পাঠক তা পাঠ করে। লেখকের হাতে শুধু শব্দ থাকে।
বাংলা ভাষা আমার মাতৃভাষা। আমি বাংলায় কথা বলি, বাংলায় লিখি, বাংলায় স্বপ্ন দেখি।
একুশে ফেব্রুয়ারি আমাদের ভাষা আন্দোলনের দিন। শহীদদের স্মরণে আমরা শ্রদ্ধা জানাই।
পৃথিবীতে যত সুন্দর জিনিস আছে, তার সবকিছুই প্রকৃতির দান। আমরা তা রক্ষা করতে হবে।
শিক্ষাই জাতির মেরুদণ্ড। শিক্ষা ছাড়া কোনো জাতি উন্নতি করতে পারে না।
মানুষ তার স্বপ্ন দিয়ে বাঁচে। স্বপ্ন না থাকলে জীবন অর্থহীন হয়ে যায়।
বই মানুষের সবচেয়ে ভালো বন্ধু। বই পড়ে আমরা জ্ঞান অর্জন করি।
সময় এবং স্রোত কারো জন্য অপেক্ষা করে না। তাই সময়ের সদ্ব্যবহার করা উচিত।
পরিশ্রম সৌভাগ্যের প্রসূতি। পরিশ্রম ছাড়া সাফল্য লাভ করা যায় না।
সততাই সর্বোত্তম পন্থা। সত্যবাদী মানুষ সবার কাছে প্রিয় হয়।
""" * 100  # Repeat to increase corpus size

def create_training_data():
    """Create training text files."""
    output_dir = Path("output/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create English training file
    en_file = output_dir / "train_en.txt"
    with open(en_file, 'w', encoding='utf-8') as f:
        f.write(ENGLISH_SAMPLES)
    
    # Create Bengali training file
    bn_file = output_dir / "train_bn.txt"
    with open(bn_file, 'w', encoding='utf-8') as f:
        f.write(BENGALI_SAMPLES)
    
    # Create combined training file
    combined_file = output_dir / "train_combined.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(ENGLISH_SAMPLES + "\n\n" + BENGALI_SAMPLES)
    
    logger.info(f"Created training files in {output_dir}")
    return combined_file

def train_tokenizer():
    """Train SentencePiece tokenizer."""
    logger.info("Training Bemation tokenizer...")
    
    # Create training data
    training_file = create_training_data()
    
    output_dir = Path("output/tokenizer")
    model_prefix = str(output_dir / "multilingual_book")
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=str(training_file),
        model_prefix=model_prefix,
        vocab_size=40000,
        character_coverage=0.9995,
        model_type='unigram',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        normalization_rule_name='nfkc',
    )
    
    logger.info(f"Tokenizer trained successfully!")
    logger.info(f"Model files saved: {model_prefix}.model and {model_prefix}.vocab")
    
    return Path(f"{model_prefix}.model")

def validate_tokenizer(model_path):
    """Validate tokenizer efficiency."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    logger.info("\n=== Tokenizer Validation ===")
    logger.info(f"Vocabulary size: {sp.vocab_size()}")
    
    # Test English
    en_text = "Once upon a time, there was a little girl who loved to read."
    en_tokens = sp.encode(en_text)
    en_words = len(en_text.split())
    en_tpw = len(en_tokens) / en_words
    
    logger.info(f"\nEnglish test:")
    logger.info(f"  Text: {en_text}")
    logger.info(f"  Tokens: {en_tokens}")
    logger.info(f"  Words: {en_words}, Tokens: {len(en_tokens)}")
    logger.info(f"  Tokens/word: {en_tpw:.2f}")
    
    # Test Bengali
    bn_text = "আমি বাংলাদেশ থেকে এসেছি। আমি একটি ভাষা মডেল তৈরি করছি।"
    bn_tokens = sp.encode(bn_text)
    bn_words = len(bn_text.split())
    bn_tpw = len(bn_tokens) / bn_words
    
    logger.info(f"\nBengali test:")
    logger.info(f"  Text: {bn_text}")
    logger.info(f"  Tokens: {bn_tokens}")
    logger.info(f"  Words: {bn_words}, Tokens: {len(bn_tokens)}")
    logger.info(f"  Tokens/word: {bn_tpw:.2f}")
    
    logger.info(f"\n✅ Tokenizer validation complete!")

if __name__ == "__main__":
    model_path = train_tokenizer()
    validate_tokenizer(model_path)
