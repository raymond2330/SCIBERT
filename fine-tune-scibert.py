import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import re
import transformers
import accelerate

# Verify the version to be sure
print(f"Transformers version: {transformers.__version__}") 

# --- 1. Configuration ---
file_path = 'arxiv_data_grouped.csv'
title_column = 'titles'
abstract_column = 'abstracts'
MODEL_CHECKPOINT = 'allenai/scibert_scivocab_uncased'
output_model_dir = './fine-tuned-scibert-multilabel'

# --- 2. Data Loading and Cleaning ---
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("Cleaning text data...")
df[title_column] = df[title_column].apply(clean_text)
df[abstract_column] = df[abstract_column].apply(clean_text)

# --- 3. Data Preprocessing ---
first_label_index = df.columns.get_loc(abstract_column) + 1
label_columns = df.columns[first_label_index:].tolist()
df['text'] = df[title_column] + " [SEP] " + df[abstract_column]

# Convert labels to float32 numpy arrays (this ensures proper dtype)
df['labels'] = df[label_columns].values.astype(np.float32).tolist()
df_clean = df[['text', 'labels']]

print(f"Dataset shape: {df_clean.shape}")
print(f"Number of label classes: {len(label_columns)}")
print(f"Sample labels: {df_clean['labels'].iloc[0]}")

# --- 4. Tokenization ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# --- 5. Dataset Creation and Processing ---
print("Creating dataset...")
full_dataset = Dataset.from_pandas(df_clean)

print("Tokenizing dataset...")
tokenized_dataset = full_dataset.map(tokenize_function, batched=True)

# Remove text column (no longer needed)
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# Set format to torch tensors (this handles the float32 conversion automatically)
print("Setting dataset format to PyTorch tensors...")
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Verify the conversion worked
sample = tokenized_dataset[0]
print(f"Labels type: {type(sample['labels'])}")
print(f"Labels dtype: {sample['labels'].dtype}")
print(f"Sample labels: {sample['labels']}")

# --- 6. Train/Test Split ---
print("Splitting dataset...")
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 7. Model Loading ---
print("Loading model...")
num_labels = len(label_columns)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    problem_type="multi_label_classification",
    num_labels=num_labels
)

# --- 8. Metrics Function ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Convert to probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    # Convert to predictions (threshold = 0.5)
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    
    # Calculate metrics
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    roc_auc = roc_auc_score(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }
    
# --- 9. Training Arguments ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    report_to=None,  # Disable wandb/tensorboard logging
    dataloader_pin_memory=False,  # Can help with memory issues
)


# --- 10. Trainer Initialization and Training ---
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping callback

)

print("\n" + "="*50)
print("STARTING MODEL TRAINING")
print("="*50)

# Train the model
trainer.train()

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)


# --- 11. Save the Model ---
print(f"Saving model to: {output_model_dir}")
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)


# --- 12. Final Evaluation ---
print("\nRunning final evaluation...")
eval_results = trainer.evaluate()

print("\nFinal Evaluation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print(f"\nModel saved successfully to: {output_model_dir}")
print("Training pipeline completed!")