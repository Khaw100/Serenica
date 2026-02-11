import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from labels import GOEMOTIONS, label2id
from config import TEXT_COL


def load_tsv(path):
    return pd.read_csv(path, sep="\t")


def parse_labels(label_str):
    return [int(x) for x in label_str.split(",")]


def id_to_label_name(label_ids):
    return [GOEMOTIONS[i] for i in label_ids]


def prepare_dataframe(df):
    df["labels_list"] = df["labels"].apply(parse_labels)
    df["labels_list"] = df["labels_list"].apply(id_to_label_name)
    return df


def make_dataset(df):
    ds = Dataset.from_pandas(df[[TEXT_COL, "labels_list"]].copy())

    def to_multihot(example):
        vec = np.zeros(len(GOEMOTIONS), dtype=np.float32)
        for l in example["labels_list"]:
            vec[label2id[l]] = 1.0
        example["labels"] = vec.tolist()
        return example

    ds = ds.map(to_multihot)
    return ds


def fix_labels(example):
    example["labels"] = [float(x) for x in example["labels"]]
    return example
