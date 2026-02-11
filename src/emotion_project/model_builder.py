import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from labels import NUM_LABELS, id2label, label2id


def build_model(model_name):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device
