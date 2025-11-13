"""
Train object detection model using TF Object Detection API
"""
import os
import tensorflow as tf
from object_detection import model_lib_v2

# Configuration
PIPELINE_CONFIG_PATH = "pipeline.config"
MODEL_DIR = "training_output"
NUM_TRAIN_STEPS = 50000
CHECKPOINT_EVERY_N = 1000

def train_model():
    """Train the object detection model."""
    
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Train
    print("Starting training...")
    model_lib_v2.train_loop(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        train_steps=NUM_TRAIN_STEPS,
        use_tpu=False,
        checkpoint_every_n=CHECKPOINT_EVERY_N,
        checkpoint_max_to_keep=5
    )
    
    print(f"Training complete! Model saved to {MODEL_DIR}")


def evaluate_model():
    """Evaluate the trained model."""
    
    print("Starting evaluation...")
    model_lib_v2.eval_continuously(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        train_steps=NUM_TRAIN_STEPS,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=5,
        checkpoint_dir=MODEL_DIR,
        wait_interval=180,
        timeout=3600
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_model()
    else:
        train_model()