import torch
from torch import nn
from tqdm import tqdm

from src.dataset import get_dataloader
from src.model import InputMethodModel

import config


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(dataloader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader()

    # 加载词表
    with open(config.PROCESSED_DATA_DIR / 'vocab.txt', 'r') as f:
        vocab_list = [line[:-1] for line in f.readlines()]

    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print('=====Epoch {}/{}====='.format(epoch + 1, config.EPOCHS))
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)
        print(f'Loss: {avg_loss}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_PATH / 'model.pt')
            print('模型保存成功')
        else:
            print('无需保存')


if __name__ == '__main__':
    train()
