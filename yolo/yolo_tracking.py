from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO("yolo11x-seg.pt")
model.to("mps")
results = model.track(source="./output_video.mp4", conf=0.3, show=True)