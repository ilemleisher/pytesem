import json
import numpy as np
import tensorflow as tf

path = '/home/ilemleisher/cnn/training/'+str(name)+'/'

MODEL_PATH = path+"cnn.keras"
MU_PATH = path+"mu.npy"
SIGMA_PATH = path+"sigma.npy"
THRESH_PATH = path+"threshold.json"

model = tf.keras.models.load_model(MODEL_PATH)
mu = np.load(MU_PATH)          
sigma = np.load(SIGMA_PATH)    

with open(THRESH_PATH, "r") as f:
    threshold = float(json.load(f).get("threshold", 0.5))

n_bins = mu.shape[1]

#Preprocessing function, edit based on datastream
def preprocess_asd(asd_batch: np.ndarray) -> np.ndarray:
    """
    asd_batch shape:
      - (n_bins,) for one sample, or
      - (N, n_bins) for batch
    returns shape: (N, n_bins, 1)
    """
    asd_batch = np.asarray(asd_batch, dtype=np.float32)
    if asd_batch.ndim == 1:
        asd_batch = asd_batch[None, :]  # -> (1, n_bins)

    if asd_batch.shape[1] != n_bins:
        raise ValueError(f"Expected {n_bins} bins, got {asd_batch.shape[1]}")

    x = np.log10(asd_batch + 1e-12)
    x = (x - mu) / (sigma + 1e-8)
    x = x[..., None]  # -> (N, n_bins, 1)
    return x.astype(np.float32)


#Inference function
def predict_asd(asd_batch: np.ndarray):
    """
    Returns logits, probabilities, binary flags
    """
    x = preprocess_asd(asd_batch)
    logits = model.predict(x, verbose=0).squeeze()
    probs = tf.sigmoid(logits).numpy()
    flags = (probs > threshold).astype(np.int32)
    return logits, probs, flags