from ultralytics import YOLOv10
import os
import torch

if __name__ == '__main__':
  
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    

    model = YOLOv10('./ultralytics/cfg/models/v10/yolov10nq2.yaml')
    


    if hasattr(model, 'fuse'):
        model.fuse()  
        print("模型已重參數化")

       
        for param in model.parameters():
            param.detach_()
            param.requires_grad = False

    model.eval()

    with torch.no_grad():
        model.val(
            data='./data_cfg/dataset.yaml',  
            batch=32, 
            imgsz=640,  
            device=0  
        )
