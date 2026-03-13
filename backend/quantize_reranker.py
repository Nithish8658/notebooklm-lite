import os
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_reranker():
    model_dir = Path(__file__).parent / "models" / "reranker-onnx"
    input_model = model_dir / "model.onnx"
    output_model = model_dir / "model_quantized.onnx"

    if not input_model.exists():
        print(f"Error: Original model not found at {input_model}")
        return

    print(f"Starting Dynamic INT8 Quantization: {input_model}...")
    
    # Dynamic quantization is the standard for Transformers (like Cross-Encoders)
    # It quantizes the weights to INT8 but keeps activations as FP32/INT8 dynamically.
    # This maintains ~99%+ accuracy for these models.
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8
    )

    print(f"SUCCESS: Quantized model saved to {output_model}")
    
    original_size = os.path.getsize(input_model) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model) / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

if __name__ == "__main__":
    quantize_reranker()
