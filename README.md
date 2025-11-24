# Bemation Tokenizer

A custom SentencePiece tokenizer optimized for English and Bengali, achieving **58% better efficiency** for Bengali text compared to GPT-2's tokenizer.

## Performance

| Tokenizer | English (tokens/word) | Bengali (tokens/word) |
|-----------|----------------------|----------------------|
| GPT-2     | ~1.3                 | ~2.5                 |
| Bemation  | ~1.15                | ~1.32                |

**Result:** Bengali text requires 47% fewer tokens, enabling faster training, longer context windows, and lower API costs.

## Installation

```bash
pip install sentencepiece
```

## Quick Start

```python
import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Encode text
tokens = sp.encode("আমি বাংলা ভালোবাসি", out_type=int)
print(f"Tokens: {tokens}")

# Decode back
text = sp.decode(tokens)
print(f"Text: {text}")

# English example
english_tokens = sp.encode("Once upon a time", out_type=int)
print(f"English tokens: {english_tokens}")
```

## Training Details

- **Vocabulary size:** 40,000
- **Training corpus:** 1,000 books (500 English, 500 Bengali)
- **Sources:** Project Gutenberg, Internet Archive, Wikisource Bengali
- **Algorithm:** Unigram language model
- **Special tokens:** `<pad>`, `<s>`, `</s>`, `<unk>`

## Why This Matters

GPT-2's tokenizer was trained primarily on English text, making it inefficient for Bengali:
- Bengali words get split into many subword tokens
- Wastes context window space
- Slower training and inference
- Higher API costs

Bemation tokenizer was trained on balanced English-Bengali corpus:
- Bengali words tokenize naturally
- 47% fewer tokens for same text
- Better for multilingual models

## Benchmarks

Compare against GPT-2 tokenizer:

```bash
python benchmark.py
```

## Use Cases

- Training multilingual models with English + Bengali
- Fine-tuning existing models on Bengali data
- Building Bengali NLP applications
- API cost optimization for Bengali text

## Training Your Own

If you want to train the tokenizer yourself:

```bash
python train_tokenizer.py
```

This will:
1. Download sample books from public domain sources
2. Create balanced English-Bengali training corpus
3. Train SentencePiece model with 40K vocabulary
4. Validate tokenization efficiency
5. Save `tokenizer.model` and `tokenizer.vocab`

## About

Built by **Inan** (13, Bangladesh) as part of the Bemation multilingual language model project. 

Training a transformer from scratch taught me that tokenization efficiency matters enormously for underserved languages.

## License

MIT License - Free to use for any purpose

## Citation

If you use this tokenizer in your research:

```bibtex
@misc{bemation-tokenizer-2025,
  author = {Inan},
  title = {Bemation Tokenizer: Efficient Multilingual Tokenization for English and Bengali},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/inanxr/BemationTokenizer}
}
```

## Coming Soon

- Full Bemation language model (100M parameters)
- Vision-language variant
- Support for more South Asian languages

---

**Star ⭐ this repo if you find it useful!**
