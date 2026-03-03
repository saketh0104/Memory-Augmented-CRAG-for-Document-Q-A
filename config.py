import os
import logging

def suppress_warnings():
    # Suppress HuggingFace + Transformers logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    # Disable tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
