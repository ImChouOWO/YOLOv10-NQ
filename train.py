from ultralytics import YOLOv10
import os
import torch

if __name__ == '__main__':

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()

    model = YOLOv10('./ultralytics/cfg/models/v10/yolov10nq2.yaml')
   

   

    model.train(
        data='./data_cfg/dataset.yaml', 
        epochs=300,    
        batch=64,  
        imgsz=640,  
        device=0,  
        save=True,  
        save_period=20,  
        plots =True,
        resume=True,
        patience=0
       
    )

