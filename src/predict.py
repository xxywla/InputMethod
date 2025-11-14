import torch

import config
from src.model import InputMethodModel
from src.tokenizer import JiebaTokenizer


def predict(text, model, tokenizer, device):
    input_index_list = tokenizer.encode(text)
    input_tensor = torch.tensor([input_index_list]).to(device)

    model.eval()
    with torch.no_grad():
        # output.shape is [batch_size, vocab_size]
        output = model(input_tensor)
        top5_index_tensor = torch.topk(output, 5).indices
        top5_indexes = top5_index_tensor.tolist()

    top5_word = [tokenizer.index2word[index] for index in top5_indexes[0]]
    print(top5_word)


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    model = InputMethodModel(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH / 'model.pt'))

    print('请输入内容（输入 q 或 quit 退出）')
    pre_input = ''
    while True:
        cur_input = input('> ')
        if cur_input == 'q' or cur_input == 'quit':
            break
        if cur_input.strip() == '':
            continue
        pre_input += cur_input
        predict(pre_input, model, tokenizer, device)


if __name__ == '__main__':
    run_predict()
