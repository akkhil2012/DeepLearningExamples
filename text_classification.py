import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, num_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc  = nn.Linear(emb_dim, num_classes)

    def forward(self, text, offsets):
        # EmbeddingBag-like manual mean pooling
        emb = self.emb(text)  # [N_tokens, emb_dim]
        # build segments from offsets
        pooled = []
        for i in range(len(offsets)):
            start = offsets[i]
            end = offsets[i+1] if i+1 < len(offsets) else emb.size(0)
            pooled.append(emb[start:end].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)
        return self.fc(pooled)

def collate_batch(batch, vocab):
    labels, tokens, offsets = [], [], [0]
    for (label, text) in batch:
        labels.append(label - 1)  # AG_NEWS labels 1..4
        t = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        tokens.append(t)
        offsets.append(offsets[-1] + t.numel())
    labels = torch.tensor(labels, dtype=torch.long)
    tokens = torch.cat(tokens)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    return tokens, offsets, labels

def main():
    train_iter = AG_NEWS(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter = list(AG_NEWS(split="train"))
    test_iter  = list(AG_NEWS(split="test"))

    train_dl = DataLoader(train_iter, batch_size=64, shuffle=True,
                          collate_fn=lambda b: collate_batch(b, vocab))
    test_dl  = DataLoader(test_iter, batch_size=256, shuffle=False,
                          collate_fn=lambda b: collate_batch(b, vocab))

    model = TextClassifier(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def acc(dl):
        model.eval()
        c, t = 0, 0
        for text, offsets, y in dl:
            text, offsets, y = text.to(DEVICE), offsets.to(DEVICE), y.to(DEVICE)
            p = model(text, offsets).argmax(1)
            c += (p == y).sum().item()
            t += y.numel()
        return c/t

    for epoch in range(3):
        model.train()
        for text, offsets, y in train_dl:
            text, offsets, y = text.to(DEVICE), offsets.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(text, offsets), y)
            loss.backward()
            opt.step()
        print("epoch", epoch, "test_acc", acc(test_dl))

if __name__ == "__main__":
    main()
