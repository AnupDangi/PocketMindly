import onnxruntime
import os
import numpy as np

model_path = "models/onnx/model.onnx"

print(f"Loading {model_path}...")
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("\nInputs:")
for i in session.get_inputs():
    print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

print("\nOutputs:")
for o in session.get_outputs():
    print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
