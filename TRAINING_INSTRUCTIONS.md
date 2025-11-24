# Bemation Tokenizer

This repository contains the trained SentencePiece tokenizer for the Bemation project.

## Note

The actual tokenizer model files (`tokenizer.model` and `tokenizer.vocab`) will be added after training.

To train the tokenizer:
1. Install dependencies: `pip install -r requirements.txt`
2. Run training script: `python train_tokenizer.py`
3. The model files will be generated in the `output/tokenizer/` directory

Then copy `output/tokenizer/multilingual_book.model` and `output/tokenizer/multilingual_book.vocab` to this directory and rename them to `tokenizer.model` and `tokenizer.vocab`.

After that, test the tokenizer with:
```bash
python example.py
python benchmark.py
```
