# src/utils.py
import joblib
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix # Import specific type if checking


from .config import MODEL_DIR, RESULTS_DIR

def save_model(model, model_name_prefix="ae_model"):
    """Saves a trained model to the configured model directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name_prefix}_{timestamp}.joblib"
    filepath = os.path.join(MODEL_DIR, filename)
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def load_model(filepath):
    """Loads a model from the specified path."""
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from: {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_results(metrics, results_name_prefix="eval_metrics"):
     """Saves evaluation metrics dictionary to a JSON file."""
     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     filename = f"{results_name_prefix}_{timestamp}.json"
     filepath = os.path.join(RESULTS_DIR, filename)

     # Convert numpy arrays/objects to lists for JSON serialization
     serializable_metrics = {}
     for key, value in metrics.items():
          if isinstance(value, np.ndarray):
               serializable_metrics[key] = value.tolist()
          elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)): # Handle numpy integers
              serializable_metrics[key] = int(value)
          elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)): # Handle numpy floats
              serializable_metrics[key] = float(value)
          elif isinstance(value, (np.bool_)): # Handle numpy bool
              serializable_metrics[key] = bool(value)
          elif isinstance(value, (pd.DataFrame, pd.Series)): # Convert pandas objects
               serializable_metrics[key] = value.to_dict() # Or use to_json()
          # Check for confusion_matrix which might be a numpy array
          elif key == 'confusion_matrix' and isinstance(value, np.ndarray):
               serializable_metrics[key] = value.tolist()
          else:
               # Attempt to keep others if they are JSON serializable by default
               try:
                   json.dumps(value) # Test serialization
                   serializable_metrics[key] = value
               except TypeError:
                   print(f"Warning: Skipping non-serializable key '{key}' of type {type(value)}")
                   serializable_metrics[key] = str(value) # Store string representation as fallback


     try:
          with open(filepath, 'w', encoding='utf-8') as f:
               json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
          print(f"Evaluation results saved successfully to: {filepath}")
          return filepath
     except TypeError as e:
          print(f"Error serializing results to JSON: {e}. Check non-serializable types.")
          return None
     except Exception as e:
          print(f"Error saving results: {e}")
          return None
