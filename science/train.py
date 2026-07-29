import numpy as np
import tensorflow as tf
import os, json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from model import cnn
from utils import get_files, filter_files, stitch_files

# Path to preprocessed data (assumes format from preprocess.py)
path = "/home/ilemleisher/data/"

# Dataset
target = 'I4_D20250102_T230851'

# Load continuous data from preprocessed files following naming format from preprocess.py
filenames = filter_files(get_files(path),target)
containers = stitch_files(path,filenames,'freqs_list','asd_list')
freqs_list = containers['freqs_list']
asd_list = containers['asd_list']

# Load corresponding pseudo label files
path += "labels/"
with np.load(path+"pca_labels_"+str(target)+".npz") as d:
    pca_labels = d['pca_labels']
    ema_labels = d['ema_labels']

# Merge labels
y_win = pca_labels+ema_labels

print(f"Number of anomalous labels: {len(np.where(y_win != 0)[0])}")

# Define total number of chunks, number of frequency bins per chunk, and number of chunks per patch
t, n_bins, k = len(freqs_list), len(freqs_list[0]), 6

X = np.log10(asd_list + 1e-12).astype(np.float32)

# Create an array of ASD data chunk patches
X_patch = np.array([X[i:i+k] for i in range(t-k+1)], dtype=np.float32)  # (N,k,n_bins)
X_patch = X_patch[..., np.newaxis]  # (N,k,n_bins,1)


# Create an array of binary labels for each patch (if any chunk in the patch is anomalous, the whole patch is labeled 
# as such)
y_patch = np.array([int(np.any(y_win[i:i+k] == 1)) for i in range(t - k + 1)], dtype=np.int32)

# Split data
N = len(X_patch)
n_train = int(0.7 * N)
n_val = int(0.15 * N)
gap = k - 1

train_end = n_train
val_start = train_end + gap
val_end = val_start + n_val
test_start = val_end + gap

X_train, y_train = X_patch[:train_end], y_patch[:train_end]
X_val, y_val = X_patch[val_start:val_end], y_patch[val_start:val_end]
X_test, y_test = X_patch[test_start:], y_patch[test_start:]

# Create unique training directory
base_dir = '/home/ilemleisher/training'
training_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(training_dir, exist_ok=True)

# Calculate references stats for normalization
mu = X_train.mean(axis=(0,1), keepdims=True)
sigma = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mu) / sigma
X_val   = (X_val - mu) / sigma
X_test  = (X_test - mu) / sigma

# Define number of epochs for training
n_epochs = 40

# Load model
model = cnn(k=k, n_bins=n_bins)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),                        #adapts LR, parameter is starting LR
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),       #uses numerically stable sigmoid + BCE formulation
    metrics=[
        tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="acc"),  #logits threshold
        tf.keras.metrics.AUC(from_logits=True, name="auc"),          #area under ROC curve
    ],
)

model.summary()

# Weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)  #assigns larger weight to rare classes
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

#Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(                             #stop if no improvement in patience epochs
        monitor="val_auc", mode="max", patience=20, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(                                             #saves model when improves
        os.path.join(training_dir, "best_asd_cnn.keras"), monitor="val_auc", mode="max", save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(                               #reduces LR by factor if no loss decrease
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )
]

#Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=n_epochs,
    batch_size=16,                   #model sees n patches and computes average loss, does one optimizer
    class_weight=class_weight,      
    callbacks=callbacks,
    verbose=1,                              #amount of training output printed per fit()
)

#Evaluation
best_model = tf.keras.models.load_model(os.path.join(training_dir, "best_asd_cnn.keras"))

val_results = best_model.evaluate(X_val, y_val, verbose=0)
test_results = best_model.evaluate(X_test, y_test, verbose=0)

val_metrics = dict(zip(best_model.metrics_names, val_results))
test_metrics = dict(zip(best_model.metrics_names, test_results))

print("Validation metrics:", val_metrics)
print("Test metrics:", test_metrics)

np.save(os.path.join(training_dir, "mu.npy"), mu)
np.save(os.path.join(training_dir, "sigma.npy"), sigma)

with open(os.path.join(training_dir, "history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

summary = {
    "val": val_metrics,
    "test": test_metrics,
}
with open(os.path.join(training_dir, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(training_dir, "threshold.json"), "w") as f:
    json.dump({"threshold": 0.5}, f, indent=2)

# Plot training and validation curves
history_dict = history.history
epochs = range(1, len(next(iter(history_dict.values()))) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history_dict["loss"], label="train loss")
axes[0].plot(epochs, history_dict["val_loss"], label="val loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss curve")
axes[0].legend()

axes[1].plot(epochs, history_dict["acc"], label="train acc")
axes[1].plot(epochs, history_dict["val_acc"], label="val acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(training_dir, "training_curves.png"), dpi=200)
plt.close(fig)

# Plot validation vs test metrics as a simple comparison bar chart
metric_names = ["loss", "acc", "auc"]
val_values = [val_metrics.get(name, np.nan) for name in metric_names]
test_values = [test_metrics.get(name, np.nan) for name in metric_names]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(metric_names))
width = 0.35
ax.bar(x - width/2, val_values, width, label="val")
ax.bar(x + width/2, test_values, width, label="test")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel("Metric value")
ax.set_title("Validation vs Test Metrics")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(training_dir, "val_test_metrics.png"), dpi=200)
plt.close(fig)

print(training_dir)