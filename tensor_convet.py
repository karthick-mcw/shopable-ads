from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./best.pt")

# Export the model to TensorRT format
engine_path = model.export(format="engine" , half=True )  

print(f"TensorRT engine saved at: {engine_path}")
