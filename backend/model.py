# ============================================================================
# OPTIMIZED ROBERTA INTENT CLASSIFICATION PIPELINE FOR T4 GPU
# ============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Check GPU availability
import torch
import subprocess

print("="*70)
print("🔍 SYSTEM DIAGNOSTICS")
print("="*70)

# GPU Info
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✅ PyTorch Version: {torch.__version__}")
else:
    print("❌ No GPU detected! Please enable GPU:")
    print("   Runtime → Change runtime type → Hardware accelerator → GPU (T4)")
    raise SystemError("GPU not available!")

# Check if T4 GPU (optimal for mixed precision)
gpu_name = torch.cuda.get_device_name(0)
if "T4" in gpu_name:
    print("✅ T4 GPU detected - Optimized for mixed precision training!")
else:
    print(f"⚠️  Detected {gpu_name} - Pipeline will adapt accordingly")

print("\n💾 Available RAM:", subprocess.check_output(['free', '-h']).decode().split('\n')[1].split()[1])
print("="*70)
# ============================================================================
# INSTALL OPTIMIZED PACKAGES
# ============================================================================

print("\n📦 Installing optimized packages...")



# Verify installations
import transformers
import datasets
import accelerate

print(f"\n📚 Package Versions:")
print(f"  - Transformers: {transformers.__version__}")
print(f"  - Datasets: {datasets.__version__}")
print(f"  - Accelerate: {accelerate.__version__}")
# ============================================================================
# UPLOAD AND VALIDATE TRAINING DATA
# ============================================================================

from google.colab import files
import pandas as pd
import io

print("\n📤 Upload your training data (airline_intents_data.csv)")
print("="*70)

uploaded = files.upload()

# Load and validate data
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

print(f"\n✅ Data loaded successfully!")
print(f"📊 Dataset Statistics:")
print(f"  - Total samples: {len(df)}")
print(f"  - Unique intents: {df['label'].nunique()}")
print(f"  - Average text length: {df['text'].str.len().mean():.1f} characters")

# Check class distribution
print(f"\n📈 Class Distribution:")
class_counts = df['label'].value_counts()
print(class_counts)

# Identify imbalanced classes
min_samples = class_counts.min()
max_samples = class_counts.max()
imbalance_ratio = max_samples / min_samples

if imbalance_ratio > 3:
    print(f"\n⚠️  WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
    print("   Consider adding more examples for minority classes")
else:
    print(f"\n✅ Class distribution is balanced (ratio: {imbalance_ratio:.2f})")

# Show sample data
print(f"\n📝 Sample Data:")
print(df.head(3))
# ============================================================================
# OPTIMIZED DATA PREPARATION WITH CACHING
# ============================================================================

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np

print("\n🔧 Preparing dataset with optimizations...")

# Create label mappings
unique_labels = sorted(df['label'].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

num_labels = len(unique_labels)
print(f"✅ Label mappings created for {num_labels} classes")

# Convert labels
df['label_id'] = df['label'].map(label2id)

# Stratified split with validation set
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'].tolist(),
    df['label_id'].tolist(),
    test_size=0.3,
    random_state=42,
    stratify=df['label_id']
)

# Further split temp into validation and test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,  # 15% validation, 15% test
    random_state=42,
    stratify=temp_labels
)

# Create datasets
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print(f"\n✅ Dataset splits created:")
print(f"  - Training: {len(train_dataset)} samples (70%)")
print(f"  - Validation: {len(val_dataset)} samples (15%)")
print(f"  - Test: {len(test_dataset)} samples (15%)")
# ============================================================================
# OPTIMIZED TOKENIZATION WITH CACHING
# ============================================================================

from transformers import AutoTokenizer

print("\n🔤 Loading and optimizing tokenizer...")

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Analyze text lengths to determine optimal max_length
text_lengths = df['text'].str.split().str.len()
p95_length = int(np.percentile(text_lengths, 95))
optimal_max_length = min(128, p95_length + 10)  # Add buffer

print(f"✅ Text length analysis:")
print(f"  - Mean: {text_lengths.mean():.1f} tokens")
print(f"  - 95th percentile: {p95_length} tokens")
print(f"  - Optimal max_length: {optimal_max_length}")

def tokenize_function(examples):
    """Optimized tokenization with padding and truncation"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=optimal_max_length,
        return_tensors=None  # Don't convert to tensors yet for efficiency
    )

# Tokenize with caching and multiple processes
print("\n🔄 Tokenizing dataset (with caching)...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # Process in large batches
    num_proc=2,  # Use 2 processes for parallel processing
    remove_columns=['text'],  # Remove text to save memory
    desc="Tokenizing"
)

# Set format for PyTorch
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print("✅ Tokenization complete with caching enabled!")
print(f"📦 Dataset size in memory: {tokenized_dataset['train'].data.nbytes / 1e6:.2f} MB")
# ============================================================================
# LOAD MODEL WITH MEMORY OPTIMIZATIONS
# ============================================================================

from transformers import AutoModelForSequenceClassification
import gc

print("\n🤖 Loading RoBERTa model with optimizations...")

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Load model with optimized settings
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.float16,  # Load in FP16 for memory efficiency
)

# Move to GPU
model = model.to('cuda')

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model loaded successfully!")
print(f"📊 Model Statistics:")
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")
print(f"  - Model size: {total_params * 2 / 1e9:.2f} GB (FP16)")

# Check GPU memory
print(f"\n💾 GPU Memory:")
print(f"  - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  - Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
# ============================================================================
# OPTIMIZED TRAINING CONFIGURATION FOR T4 GPU
# ============================================================================

from transformers import TrainingArguments, DataCollatorWithPadding

print("\n⚙️ Configuring optimized training parameters...")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# OPTIMIZED TRAINING ARGUMENTS FOR T4 GPU
training_args = TrainingArguments(
    output_dir='./results',
    
    # === BATCH SIZE OPTIMIZATION ===
    per_device_train_batch_size=32,  # T4 can handle this with FP16
    per_device_eval_batch_size=64,   # Larger batch for evaluation
    gradient_accumulation_steps=2,   # Effective batch size = 64
    
    # === MIXED PRECISION (KEY OPTIMIZATION) ===
    fp16=True,  # Enable mixed precision for 2x speedup
    fp16_opt_level='O2',  # Full mixed precision
    
    # === LEARNING RATE ===
    learning_rate=2e-5,
    warmup_ratio=0.1,  # 10% warmup steps
    
    # === TRAINING DURATION ===
    num_train_epochs=3,
    max_steps=-1,  # Use epochs instead
    
    # === OPTIMIZATION ===
    optim='adamw_torch',  # Optimized AdamW
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    
    # === EVALUATION ===
    eval_strategy='epoch',
    eval_steps=None,
    eval_accumulation_steps=1,  # Don't accumulate eval results
    
    # === SAVING ===
    save_strategy='epoch',
    save_total_limit=2,  # Keep only best 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    
    # === LOGGING ===
    logging_dir='./logs',
    logging_strategy='steps',
    logging_steps=10,
    report_to='none',  # Disable external logging
    
    # === DATA LOADING ===
    dataloader_num_workers=2,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster data transfer to GPU
    
    # === PERFORMANCE ===
    gradient_checkpointing=True,  # Save memory
    remove_unused_columns=True,
    label_smoothing_factor=0.1,  # Improve generalization
    
    # === REPRODUCIBILITY ===
    seed=42,
    data_seed=42,
)

print("✅ Training configuration optimized for T4 GPU!")
print(f"\n📊 Training Details:")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Total training steps: {len(tokenized_dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
print(f"  - Mixed precision: {'✅ Enabled (FP16)' if training_args.fp16 else '❌ Disabled'}")
print(f"  - Gradient checkpointing: {'✅ Enabled' if training_args.gradient_checkpointing else '❌ Disabled'}")
# ============================================================================
# OPTIMIZED METRICS COMPUTATION
# ============================================================================

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

def compute_metrics(eval_pred):
    """Compute comprehensive metrics efficiently"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted',
        zero_division=0
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

print("✅ Metrics computation configured!")
# ============================================================================
# OPTIMIZED TRAINING WITH PROGRESS MONITORING
# ============================================================================

from transformers import Trainer
import time

print("\n🏋️ Initializing Trainer with optimizations...")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized!")
print("\n" + "="*70)
print("🚀 STARTING OPTIMIZED TRAINING")
print("="*70)

# Training with timing
start_time = time.time()

try:
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"🚀 Samples per second: {len(tokenized_dataset['train']) * training_args.num_train_epochs / training_time:.2f}")
    print(f"📊 Final training loss: {train_result.training_loss:.4f}")
    
except Exception as e:
    print(f"\n❌ Training failed: {str(e)}")
    raise

# GPU Memory usage
print(f"\n💾 Final GPU Memory Usage:")
print(f"  - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  - Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"  - Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
# ============================================================================
# COMPREHENSIVE MODEL EVALUATION
# ============================================================================

print("\n📈 Evaluating model on validation set...")
val_results = trainer.evaluate(eval_dataset=tokenized_dataset['validation'])

print("\n" + "="*70)
print("📊 VALIDATION RESULTS")
print("="*70)
for key, value in val_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

print("\n📈 Evaluating model on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])

print("\n" + "="*70)
print("📊 TEST RESULTS (Unseen Data)")
print("="*70)
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Detailed classification report
print("\n📋 Generating detailed classification report...")
predictions = trainer.predict(tokenized_dataset['test'])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\n" + "="*70)
print("📊 DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(
    true_labels, 
    pred_labels, 
    target_names=[id2label[i] for i in range(num_labels)],
    digits=3
))
# ============================================================================
# TEST MODEL WITH REAL EXAMPLES
# ============================================================================

from transformers import pipeline

print("\n🧪 Testing model with real examples...")

# Create classifier pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0  # Use GPU
)

# Test messages
test_messages = [
    "I want to cancel my flight to New York",
    "My luggage is missing at the airport",
    "What is the status of flight AA123",
    "Can I travel with my pet dog",
    "How much baggage can I check in",
    "I need to file a complaint about the service",
    "Do you offer student discounts",
    "What items are not allowed on the plane"
]

print("\n" + "="*70)
print("🎯 PREDICTION RESULTS")
print("="*70)

for msg in test_messages:
    result = classifier(msg)[0]
    label_id = int(result['label'].split('_')[-1])
    predicted_intent = id2label[label_id]
    confidence = result['score']
    
    print(f"\n📝 Message: \"{msg}\"")
    print(f"✅ Prediction: {predicted_intent}")
    print(f"🎯 Confidence: {confidence:.2%}")
# ============================================================================
# SAVE MODEL FOR PRODUCTION USE
# ============================================================================

import json

print("\n💾 Saving model for production deployment...")

# Save model and tokenizer
model.save_pretrained('./airline_intent_classifier')
tokenizer.save_pretrained('./airline_intent_classifier')

# Save label mappings
with open('./airline_intent_classifier/label_mappings.json', 'w') as f:
    json.dump({
        'label2id': label2id,
        'id2label': id2label
    }, f, indent=2)

# Save training metadata
metadata = {
    'model_name': MODEL_NAME,
    'num_labels': num_labels,
    'training_samples': len(tokenized_dataset['train']),
    'validation_accuracy': val_results['eval_accuracy'],
    'test_accuracy': test_results['eval_accuracy'],
    'training_time_seconds': training_time,
    'mixed_precision': training_args.fp16,
    'batch_size': training_args.per_device_train_batch_size,
    'epochs': training_args.num_train_epochs
}

with open('./airline_intent_classifier/training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model saved successfully!")
print("\n📁 Saved files:")
print("  - Model weights: pytorch_model.bin")
print("  - Configuration: config.json")
print("  - Tokenizer files: tokenizer.json, vocab.json, merges.txt")
print("  - Label mappings: label_mappings.json")
print("  - Training metadata: training_metadata.json")
# ============================================================================
# CREATE DOWNLOADABLE PACKAGE
# ============================================================================

import shutil
from google.colab import files

print("\n📦 Creating downloadable package...")

# Create zip file
shutil.make_archive('airline_intent_classifier', 'zip', './airline_intent_classifier')

# Get file size
file_size = os.path.getsize('airline_intent_classifier.zip') / 1e6

print(f"✅ Package created!")
print(f"📦 File size: {file_size:.2f} MB")

print("\n⬇️ Downloading model package...")
files.download('airline_intent_classifier.zip')

print("\n" + "="*70)
print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📋 Next Steps:")
print("  1. Extract 'airline_intent_classifier.zip' on your local machine")
print("  2. Place the extracted folder in your backend directory")
print("  3. Update main.py to load the model")
print("  4. Restart your backend server")
print("  5. Test predictions in your frontend!")
print("\n✅ Model is production-ready!")
