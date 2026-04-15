from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding
)
from config import MODEL_NAME, TEXT_COL, MAX_LENGTH
from dataset_builder import prepare_dataframe, make_dataset, fix_labels
from model_builder import build_model
from trainer_utils import compute_metrics, compute_pos_weights, CustomTrainer
from dataset_builder import (
    load_tsv,
    prepare_dataframe,
    make_dataset,
    fix_labels
)

# Load data
train = load_tsv("data/emotion_data/train.tsv")
dev   = load_tsv("data/emotion_data/dev.tsv")

# Prepare data
train = prepare_dataframe(train)
dev   = prepare_dataframe(dev)

train_ds = make_dataset(train)
val_ds   = make_dataset(dev)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.map(fix_labels)

dataset = dataset.remove_columns(
    [c for c in [TEXT_COL, "labels_list", "__index_level_0__"]
     if c in dataset["train"].column_names]
)

# Model
model, device = build_model(MODEL_NAME)

# Pos weight
pos_weights = compute_pos_weights(dataset["train"], device)

# Trainer
training_args = TrainingArguments(
    output_dir="models/emotion_models",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    save_strategy="epoch",
    seed=42
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    pos_weights=pos_weights
)

trainer.train()

trainer.save_model("models/emotion_models")
tokenizer.save_pretrained("models/emotion_models")