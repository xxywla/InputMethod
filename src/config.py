from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_PATH = Path(__file__).parent.parent / "model"

SEQ_LEN = 5
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 5

if __name__ == '__main__':
    print(DATA_DIR)
