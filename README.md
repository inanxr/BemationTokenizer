# Bemation Tokenizer

An efficient SentencePiece tokenizer optimized for multilingual text processing.

## Features

- ✅ **Multilingual support** - Works with any language combination
- ✅ **Efficient tokenization** - Optimized for non-English languages
- ✅ **Easy to use** - Simple Python API
- ✅ **Customizable** - Train on your own corpus
- ✅ **Production-ready** - Based on proven SentencePiece technology

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
tokens = sp.encode("Your text here", out_type=int)
print(f"Tokens: {tokens}")

# Decode back
text = sp.decode(tokens)
print(f"Text: {text}")
```

## Training Your Own Tokenizer

```bash
python train_tokenizer.py --input your_data.txt --vocab-size 40000
```

See [TRAINING.md](TRAINING.md) for detailed instructions.

## Why SentencePiece?

SentencePiece is language-independent and handles text directly from raw sentences:
- No need for pre-tokenization
- Works with any language (including those without spaces)
- Handles unknown words gracefully
- Efficient subword tokenization

## Use Cases

- Training multilingual language models
- Fine-tuning models on custom domains
- Building NLP applications for underserved languages
- API cost optimization through efficient tokenization

## Examples

See `example.py` for usage examples and `benchmark.py` to measure performance.

## Technical Details

- **Algorithm:** Unigram language model
- **Default vocab:** 40,000 tokens
- **Special tokens:** `<pad>`, `<s>`, `</s>`, `<unk>`
- **Normalization:** NFKC Unicode normalization

## License

MIT License - Free for commercial and non-commercial use

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## About

Created for the Bemation project to enable efficient multilingual language modeling.

---

**Star ⭐ this repo if you find it useful!**
