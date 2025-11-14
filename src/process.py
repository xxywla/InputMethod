import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.tokenizer import JiebaTokenizer


def build_dataset(sentences, tokenizer):
    dataset_list = []
    seq_len = config.SEQ_LEN

    for sentence in sentences:
        sentence_indexes = tokenizer.encode(sentence)
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
    JiebaTokenizer.build_vocab(config.PROCESSED_DATA_DIR / "vocab.txt", train_sentences)

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "vocab.txt")

    train_dataset = build_dataset(train_sentences, tokenizer)
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / "train_dataset.jsonl", lines=True, orient="records")

    test_dataset = build_dataset(test_sentences, tokenizer)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl", lines=True, orient="records")


if __name__ == '__main__':
    process()
