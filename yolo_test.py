from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11m-seg.pt")  # Load a custom model from a local file
model.to("mps")
results = model("istockphoto-858326176-612x612.jpg")  # Predict on an image
results[0].show()  # Display results