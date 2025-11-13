const help_text = `# 1. Convert dataset to TFRecord format
python xml_to_tfrecord.py

# 2. Download pretrained model
bash download_pretrained_model.sh

# 3. Edit pipeline.config
# - Update num_classes to match your dataset
# - Update paths to your TFRecord files and label map
# - Adjust batch_size based on GPU memory
# - Update fine_tune_checkpoint path

# 4. Train the model
python train_detection.py

# 5. (Optional) Run evaluation in parallel
python train_detection.py eval

# 6. Monitor training with TensorBoard
tensorboard --logdir=training_output

# 7. Export trained model
python export_model.py

# 8. Run inference
python inference.py`;
console.log(help_text);
