import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def build_dataset(sentences, word2index):
    dataset_list = []
    seq_len = config.SEQ_LEN

    for sentence in sentences:
        sentence_indexes = [word2index.get(word, 0) for word in jieba.lcut(sentence)]
        for i in range(len(sentence_indexes) - seq_len):
            input_ids = sentence_indexes[i:i + seq_len]
            target_id = sentence_indexes[i + seq_len]
            dataset_list.append({'input': input_ids, 'target': target_id})
    return dataset_list


def process():
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True, orient="records")
    sentences = []
    for item in df['dialog']:
        sentences.extend(item)
    sentences = [sentence.split("：")[-1] for sentence in sentences]
    print(sentences[0])
    print(len(sentences))

    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # 构建词表
    vocab_set = set()
    for sentence in train_sentences:
        for word in jieba.cut(sentence):
            vocab_set.add(word)
    vocab_list = ['<unk>'] + list(vocab_set)
    print(f'词表大小{len(vocab_list)}')

    # 保存词表
    with open(config.PROCESSED_DATA_DIR / "vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab_list:
            f.write(word + "\n")
    print('词表保存完成')

    word2index = {word: i for i, word in enumerate(vocab_list)}

    train_dataset = build_dataset(train_sentences, word2index)
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / "train_dataset.jsonl", lines=True, orient="records")

    test_dataset = build_dataset(test_sentences, word2index)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl", lines=True, orient="records")


if __name__ == '__main__':
    process()
