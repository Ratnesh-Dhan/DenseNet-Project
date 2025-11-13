"""
Export trained model for inference
"""
import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def export_inference_graph(
    pipeline_config_path="pipeline.config",
    trained_checkpoint_dir="training_output",
    output_directory="exported_model"
):
    """Export trained model to SavedModel format."""
    
    # Load pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    # Export
    exporter_lib_v2.export_inference_graph(
        input_type='image_tensor',
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_directory
    )
    
    print(f"Model exported to {output_directory}")


if __name__ == "__main__":
    export_inference_graph()