"""
train_french_ats_model.py
Full pipeline to fine-tune a French resume–job similarity model.

Steps:
1. Load (or generate) resume–job dataset
2. Split into train/val/test
3. Fine-tune SentenceTransformer
4. Evaluate on validation/test
5. Save fine-tuned model
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
DATA_FILE = "resume_job_pairs_fr.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Run generate_french_ats_dataset.py first."
    )

df = pd.read_csv(DATA_FILE)
print(f"📄 Loaded {len(df)} resume–job pairs.")

# ------------------------------------------------------------
# 2. Split train/validation/test
# ------------------------------------------------------------
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"✅ Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

train_df.to_csv("train_pairs.csv", index=False, encoding="utf-8-sig")
val_df.to_csv("val_pairs.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("test_pairs.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. Convert to SentenceTransformer InputExamples
# ------------------------------------------------------------
def df_to_examples(dataframe):
    return [
        InputExample(
            texts=[row["resume_text"], row["job_description"]],
            label=float(row["score"]),
        )
        for _, row in dataframe.iterrows()
    ]

train_examples = df_to_examples(train_df)
val_examples = df_to_examples(val_df)
test_examples = df_to_examples(test_df)

print(f"🧠 Prepared {len(train_examples)} training examples.")

# ------------------------------------------------------------
# 4. Load model (multilingual or French)
# ------------------------------------------------------------
# You can replace this with another French-friendly model (like camembert)
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
print(f"🚀 Loaded base model: {model_name}")

# ------------------------------------------------------------
# 5. DataLoader + Loss
# ------------------------------------------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# ------------------------------------------------------------
# 6. Training configuration
# ------------------------------------------------------------
num_epochs = 2
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_dir = "./models/ats_fr_similarity"

print(f"🛠 Starting training for {num_epochs} epochs...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    output_path=output_dir,
)

print(f"💾 Model saved to {output_dir}")

# ------------------------------------------------------------
# 7. Evaluate on validation and test
# ------------------------------------------------------------
def evaluate_model(model, examples, label_name="Validation"):
    with torch.no_grad():
        resume_texts = [ex.texts[0] for ex in examples]
        job_texts = [ex.texts[1] for ex in examples]
        labels = torch.tensor([ex.label for ex in examples])

        emb1 = model.encode(resume_texts, convert_to_tensor=True, show_progress_bar=True)
        emb2 = model.encode(job_texts, convert_to_tensor=True, show_progress_bar=True)

        sims = util.cos_sim(emb1, emb2).diagonal()
        corr = torch.corrcoef(torch.stack([sims, labels]))[0, 1].item()
        print(f"📈 {label_name} correlation between cosine sim and labels: {corr:.3f}")
        return corr


val_corr = evaluate_model(model, val_examples, "Validation")
test_corr = evaluate_model(model, test_examples, "Test")

# ------------------------------------------------------------
# 8. Final summary
# ------------------------------------------------------------
print("✅ Training complete!")
print(f"📊 Validation correlation: {val_corr:.3f}")
print(f"📊 Test correlation: {test_corr:.3f}")
print(f"🧩 Fine-tuned model stored in: {output_dir}")
