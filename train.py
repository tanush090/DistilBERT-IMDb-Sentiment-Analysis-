
# ============================================================
# DISTILBERT SENTIMENT ANALYSIS ON IMDb REVIEWS
# Complete Training and Evaluation Pipeline
# ============================================================

import os
import random
import numpy as np
import torch

# Dataset and Transformer libraries
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

from torch.utils.data import DataLoader

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
MODEL_NAME = "distilbert/distilbert-base-uncased"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Save the best fine-tuned model locally
OUTPUT_DIR = "./DistilBERT_IMDb_Best"


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE CONFIGURATION
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected. Training will be slower.")


# ============================================================
# 3. LOAD IMDb DATASET AND SPLIT DATA
# ============================================================

print("\nLoading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

print("Original training samples:", len(dataset["train"]))
print("Original testing samples:", len(dataset["test"]))

# Split original 25K training samples into:
# 20K training samples and 5K validation samples

split = dataset["train"].train_test_split(
    test_size=5000,
    seed=SEED,
    stratify_by_column="label"
)

train_data = split["train"]
val_data = split["test"]
test_data = dataset["test"]

print("\n========== DATA SPLIT ==========")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))


# ============================================================
# 4. LOAD DISTILBERT TOKENIZER AND TOKENIZE REVIEWS
# ============================================================

print("\nLoading DISTILBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Convert raw review text into token IDs and attention masks
def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN
    )


print("\nTokenizing training data...")

train_tokenized = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenizing validation data...")

val_tokenized = val_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenizing test data...")

test_tokenized = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# 5. DATA COLLATOR AND DATALOADERS
# ============================================================

# Dynamically pad sequences within each batch
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

train_loader = DataLoader(
    train_tokenized,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    val_tokenized,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_tokenized,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=data_collator
)


# ============================================================
# 6. LOAD PRETRAINED DISTILBERT MODEL
# ============================================================

print("\nLoading pretrained DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NEGATIVE",
        1: "POSITIVE"
    },
    label2id={
        "NEGATIVE": 0,
        "POSITIVE": 1
    }
)

model.to(device)

print("Model loaded successfully!")


# ============================================================
# 7. OPTIMIZER AND LEARNING RATE SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# ============================================================
# 8. VALIDATION / EVALUATION FUNCTION
# ============================================================

def evaluate_model(loader):

    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, all_predictions, all_labels


# ============================================================
# 9. MODEL TRAINING AND BEST CHECKPOINT SAVING
# ============================================================

best_val_accuracy = 0.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n========== TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(train_loader):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (
            predictions == batch["labels"]
        ).sum().item()

        total += batch["labels"].size(0)

        if (step + 1) % 100 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct / total

    # Evaluate on validation data after each epoch
    val_loss, val_accuracy, _, _ = evaluate_model(
        val_loader
    )

    print("\n------------------------------------")
    print(f"Epoch: {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    print("------------------------------------")

    # Save checkpoint when validation accuracy improves
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("Best model checkpoint saved!")


# ============================================================
# 10. LOAD BEST MODEL CHECKPOINT
# ============================================================

print("\nLoading best model checkpoint...")

model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR
).to(device)

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

test_loss, test_accuracy, predictions, labels = (
    evaluate_model(test_loader)
)

print("Model: DistilBERT")

print(
    "Best Validation Accuracy:",
    round(best_val_accuracy * 100, 2),
    "%"
)

print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        labels,
        predictions,
        target_names=["Negative", "Positive"],
        digits=4
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        labels,
        predictions
    )
)


# ============================================================
# 12. SAVE FINAL MODEL AND TOKENIZER
# ============================================================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved at:", OUTPUT_DIR)


# ============================================================
# 13. CUSTOM REVIEW PREDICTION FUNCTION
# ============================================================

@torch.inference_mode()
def predict_sentiment(review):

    model.eval()

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = torch.argmax(probabilities).item()

    sentiment = (
        "Positive" if prediction == 1 else "Negative"
    )

    confidence = probabilities[prediction].item() * 100

    return {
        "Sentiment": sentiment,
        "Confidence": round(confidence, 2),
        "Negative": round(probabilities[0].item() * 100, 2),
        "Positive": round(probabilities[1].item() * 100, 2)
    }


# ============================================================
# 14. TEST DIFFICULT REVIEWS
# ============================================================

test_reviews = [

    "This movie was not bad at all.",

    "The acting was terrible and the story was boring.",

    "I absolutely loved this movie.",

    "The movie started well but the ending was disappointing.",

    "I expected to hate this movie, but surprisingly I loved it.",

    "The movie was neither good nor bad.",

    "I would not say this movie was disappointing.",

    "The story was amazing but the acting was not very impressive.",

    "I cannot deny that this was a wonderful movie.",

    "The movie was not particularly interesting and I regret watching it.",

    "Despite its flaws, I really enjoyed the movie.",

    "The film was not perfect, but it was definitely worth watching."
]

print("\n========== CUSTOM REVIEW TESTING ==========")

for review in test_reviews:

    result = predict_sentiment(review)

    print("\nReview:", review)
    print("Sentiment:", result["Sentiment"])
    print("Confidence:", result["Confidence"], "%")
    print("Positive:", result["Positive"], "%")
    print("Negative:", result["Negative"], "%")

print("\n========== ALL TASKS COMPLETED ==========")
