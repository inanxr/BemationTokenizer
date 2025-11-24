# Training Guide

## Preparing Your Data

1. **Collect training text** from your target languages
2. **Balance your corpus** - include roughly equal amounts of text from each language
3. **Clean your text** - remove excessive formatting, HTML, etc.
4. **Save as UTF-8** text file (one sentence per line recommended)

## Training the Tokenizer

```bash
python train_tokenizer.py --input your_data.txt --vocab-size 40000
```

### Options:

- `--input`: Your training text file (required)
- `--vocab-size`: Vocabulary size (default: 40000)
- `--model-prefix`: Output filename (default: tokenizer)
- `--model-type`: unigram, bpe, char, or word (default: unigram)
- `--character-coverage`: Coverage ratio (default: 0.9995)

## Recommendations

### Vocabulary Size
- Small models (100M params): 32,000
- Medium models (1B params): 40,000 - 50,000
- Large models (7B+ params): 64,000 - 100,000

### Character Coverage
- Single language: 0.995
- Multilingual (similar scripts): 0.9995
- Multilingual (diverse scripts): 0.99995

### Training Corpus Size
- Minimum: 1MB of text
- Recommended: 10MB - 100MB
- More diverse text = better tokenization

## Example

```bash
# Train on English corpus
python train_tokenizer.py \
  --input english_corpus.txt \
  --vocab-size 32000 \
  --model-prefix en_tokenizer

# Train on multilingual corpus
python train_tokenizer.py \
  --input multilingual_corpus.txt \
  --vocab-size 50000 \
  --character-coverage 0.9995 \
  --model-prefix multilingual_tokenizer
```

## Testing

After training, test your tokenizer:

```bash
python example.py
python benchmark.py
```

## Output Files

- `tokenizer.model` - The trained model (this is what you load)
- `tokenizer.vocab` - The vocabulary file (for reference)

## Tips

1. **Balanced corpus**: Include equal proportions of each language
2. **More data**: Generally better, but diminishing returns after ~100MB
3. **Clean data**: Remove code, HTML, excessive punctuation
4. **Test thoroughly**: Validate on held-out text from all languages
5. **Iterate**: Try different vocab sizes and compare performance
