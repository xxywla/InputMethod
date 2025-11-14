import torch

from src.dataset import get_dataloader
from src.model import InputMethodModel

import config
from src.tokenizer import JiebaTokenizer


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "vocab.txt")

    model = InputMethodModel(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH / 'model.pt'))

    dataloader = get_dataloader(is_train=False)

    model.eval()

    top1_accuracy = 0
    top5_accuracy = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        top5_index_tensor = torch.topk(outputs, 5).indices
        top5_indexes = top5_index_tensor.tolist()
        labels = labels.tolist()

        total += len(labels)
        for top5_index, label in zip(top5_indexes, labels):
            if top5_index[0] == label:
                top1_accuracy += 1
            if label in top5_index:
                top5_accuracy += 1

    print(f"Top-1 Accuracy: {top1_accuracy / total:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy / total:.4f}")


if __name__ == '__main__':
    evaluate()
