from ultralytics import YOLO

model = YOLO("yolo11m.pt")
# creates 'yolo11m.engine'
# half : True = fp16, False = fp32
model.export(format="engine",
             imgsz = 640,
             half = True)  

# Run inference
model = YOLO("yolo11m.engine")
results = model("https://ultralytics.com/images/bus.jpg")
