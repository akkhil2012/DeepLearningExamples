import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer

def tokenize_and_align(examples, tokenizer, label_all_tokens=False):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(label[wid])
            else:
                label_ids.append(label[wid] if label_all_tokens else -100)
            prev = wid
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

def main():
    ds = load_dataset("conll2003")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    num_labels = ds["train"].features["ner_tags"].feature.num_classes

    tokenized = ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"].select(range(2000)),
        eval_dataset=tokenized["validation"].select(range(500)),
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    main()
