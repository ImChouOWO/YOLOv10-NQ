from ultralytics import YOLOv10
import os
import torch

if __name__ == '__main__':
    # 避免在 Windows 上的多次導入問題
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # 初始化 YOLOv10 模型
    model = YOLOv10('C:/project/yolov10/ultralytics/cfg/models/v10/yolov10n.yaml')
    # model = YOLOv10('C:/project/yolov10/runs/detect/test_1/train/weights/best.pt')

    # 呼叫重參數化函數
    if hasattr(model, 'fuse'):
        model.fuse()  # 將模型轉換為推論模式
        print("模型已重參數化")

        # 確保所有參數使用 detach 並設置 requires_grad=False
        for param in model.parameters():
            param.detach_()
            param.requires_grad = False
    else:
        print("此模型不支援重參數化或已經重參數化")

    # 切換為推論模式
    model.eval()

    # 進行模型驗證
    with torch.no_grad():
        model.val(
            data='C:/project/yolov10/data_cfg/dataset.yaml',  # 使用相同的數據集進行驗證
            batch=32,  # 驗證批次大小
            imgsz=640,  # 驗證時的圖像大小
            device=0  # 使用 GPU 驗證
        )
