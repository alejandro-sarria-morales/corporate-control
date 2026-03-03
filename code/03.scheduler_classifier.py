#===============
# Set up
#===============

import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = tokenizer(self.texts[idx],
                           max_length=256,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        return {"input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx])}

#===============
# Training data
#===============
print("Step 1: Loading data...")
df = pd.read_csv("data/labelled_docs.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

#===============
# Tokenizer
#===============
print("Step 2: Setting up tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

train_loader = DataLoader(SimpleDataset(train_df["text"].tolist(), train_df["label"].tolist()), batch_size=16, shuffle=True)
test_loader = DataLoader(SimpleDataset(test_df["text"].tolist(), test_df["label"].tolist()), batch_size=16)

#===============
# Model
#===============
print("Step 3: Loading RoBERTa model...")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
print(f"  Using device: {device}")

#===============
# Training loop
#===============
print("Step 4: Training...")
for epoch in range(4):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch} — Loss: {avg_loss:.4f}")
#===============
# Evaluation
#===============
print("Step 5: Evaluating...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nResults:")
print(classification_report(all_labels, all_preds))

#===============
# Save model
#===============
print("Step 6: Saving model...")
model.save_pretrained("../models/roberta_corp_control")
tokenizer.save_pretrained("../models/roberta_corp_control")
print("Done!")