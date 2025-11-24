"""
Train a custom multilingual SentencePiece tokenizer.

Usage:
    python train_tokenizer.py --input data.txt --vocab-size 40000

The input file should contain your training text (one sentence per line, or paragraphs).
For multilingual tokenizers, include text from all target languages in roughly equal proportions.
"""

import argparse
import sentencepiece as spm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_tokenizer(
    input_file,
    model_prefix="tokenizer",
    vocab_size=40000,
    character_coverage=0.9995,
    model_type='unigram'
):
    """
    Train a SentencePiece tokenizer.
    
    Args:
        input_file: Path to training text file
        model_prefix: Output model name prefix
        vocab_size: Vocabulary size (default 40000)
        character_coverage: Character coverage (0.9995 recommended for multilingual)
        model_type: Model type ('unigram', 'bpe', 'char', or 'word')
    """
    logger.info(f"Training SentencePiece tokenizer...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Model type: {model_type}")
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
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
    
    logger.info(f"✅ Tokenizer training complete!")
    logger.info(f"Model saved: {model_prefix}.model")
    logger.info(f"Vocab saved: {model_prefix}.vocab")
    
    return Path(f"{model_prefix}.model")


def validate_tokenizer(model_path, test_texts):
    """
    Validate tokenizer on sample texts.
    
    Args:
        model_path: Path to trained model
        test_texts: List of test strings
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    logger.info("\n" + "="*60)
    logger.info("TOKENIZER VALIDATION")
    logger.info("="*60)
    logger.info(f"Vocabulary size: {sp.vocab_size()}")
    
    for i, text in enumerate(test_texts, 1):
        tokens = sp.encode(text, out_type=int)
        words = len(text.split())
        tpw = len(tokens) / words if words > 0 else 0
        
        logger.info(f"\nTest {i}:")
        logger.info(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        logger.info(f"  Tokens: {len(tokens)}")
        logger.info(f"  Words: {words}")
        logger.info(f"  Tokens/word: {tpw:.2f}")
    
    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Train a multilingual SentencePiece tokenizer'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input text file for training'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=40000,
        help='Vocabulary size (default: 40000)'
    )
    parser.add_argument(
        '--model-prefix',
        type=str,
        default='tokenizer',
        help='Output model name prefix (default: tokenizer)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='unigram',
        choices=['unigram', 'bpe', 'char', 'word'],
        help='Model type (default: unigram)'
    )
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='Character coverage (default: 0.9995 for multilingual)'
    )
    parser.add_argument(
        '--test-texts',
        nargs='+',
        help='Optional test texts for validation'
    )
    
    args = parser.parse_args()
    
    # Train tokenizer
    model_path = train_tokenizer(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type
    )
    
    # Validate if test texts provided
    if args.test_texts:
        validate_tokenizer(model_path, args.test_texts)


if __name__ == "__main__":
    main()
