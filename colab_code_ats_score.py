# finetune_ats_ranking.py - CORRECTED SCRIPT

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime

# --- Configuration ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# IMPORTANT: This path must point to your 100k generated CSV
DATASET_PATH = 'data/exports/resume_job_pairs_fr.csv'
OUTPUT_DIR = f'output/ats-finetuned-ranking-{datetime.now().strftime("%Y%m%d-%H%M")}'

# Hyperparameters for Ranking Loss (MNRL)
# Use 1 or 2 epochs for 100k data.
NUM_EPOCHS = 1
# IMPORTANT: Increased Batch Size (64) is crucial for Ranking Loss (more negative samples)
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 2e-5

# Calculate steps for WARMUP_STEPS
df_size = len(pd.read_csv(DATASET_PATH))
TOTAL_STEPS = int(df_size * 0.8 / TRAIN_BATCH_SIZE * NUM_EPOCHS)
WARMUP_STEPS = int(TOTAL_STEPS * 0.1)
EVAL_STEPS = 2000 # Evaluate every 2000 steps

# --- 1. Load and Prepare Data ---

def prepare_data(data_path):
    """Loads the dataset and splits it into train/val/test sets."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Split: 80% Train, 10% Validation, 10% Test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Total Pairs: {len(df)} | Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    # Convert DataFrames to a list of InputExample objects
    train_examples = [
        InputExample(texts=[row['resume_text'], row['job_description']], label=float(row['score']))
        for index, row in train_df.iterrows()
    ]
    return train_examples, val_df, test_df

# --- 2. Setup Model, Loss, and Dataloader (CORRECTED FUNCTION) ---

def setup_training(train_examples, val_df):
    """Initializes the model, dataset, and Ranking loss function."""
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # **FIX APPLIED HERE:** Pass the list of InputExample objects directly to DataLoader.
    # The MultipleNegativesRankingLoss handles the internal SentencesDataset conversion.
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=TRAIN_BATCH_SIZE
    )

    # RANKING LOSS: The solution to low Spearman correlation
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluation: Use the EmbeddingSimilarityEvaluator to track Spearman
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=val_df['resume_text'].tolist(),
        sentences2=val_df['job_description'].tolist(),
        scores=val_df['score'].tolist(),
        name='sts-dev'
    )

    return model, train_dataloader, train_loss, evaluator

# --- 3. Fine-Tuning ---

def fine_tune(model, train_dataloader, train_loss, evaluator):
    """Starts the fine-tuning process."""
    print("\n--- Starting Fine-Tuning with Ranking Loss ---")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {TRAIN_BATCH_SIZE}, Total Steps: {TOTAL_STEPS}")
    print(f"Warmup Steps: {WARMUP_STEPS}, Evaluate Every: {EVAL_STEPS} steps.")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_DIR,
        save_best_model=True,  # Saves the model that performs best on the validation set (highest Spearman)
        monitor_after_epoch=0, # Start monitoring immediately
        optimizer_params={'lr': LEARNING_RATE}
    )
    print(f"\n✅ Fine-tuning complete. Best ranking model saved to: {OUTPUT_DIR}")

# --- 4. Final Evaluation ---

def final_evaluation(test_df):
    """Loads the best model and evaluates it on the hold-out test set."""
    print("\n--- Final Evaluation on Test Set ---")
    best_model = SentenceTransformer(OUTPUT_DIR)

    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_df['resume_text'].tolist(),
        sentences2=test_df['job_description'].tolist(),
        scores=test_df['score'].tolist(),
        name='sts-test'
    )

    results = test_evaluator(best_model)
    print("\nFinal Test Set Results:")
    print(f"  Spearman Correlation: {results['sts-test_Spearman_Correlation']:.4f}")
    print(f"  Pearson Correlation:  {results['sts-test_Pearson_Correlation']:.4f}")

# --- Main Execution ---

if __name__ == '__main__':
    train_ex, val_df_data, test_df_data = prepare_data(DATASET_PATH)
    model, train_loader, train_loss, evaluator = setup_training(train_ex, val_df_data)
    fine_tune(model, train_loader, train_loss, evaluator)
    final_evaluation(test_df_data)