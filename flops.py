from ultralytics import YOLOv10

model = YOLOv10('C:/project/yolov12/ultralytics/cfg/models/v12/yolov12.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()