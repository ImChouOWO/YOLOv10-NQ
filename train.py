from ultralytics import YOLOv10
import os
import torch

if __name__ == '__main__':
    # 避免在 Windows 上的多次導入問題
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    # 初始化 YOLOv10 模型
    model = YOLOv10('C:/project/yolov10/ultralytics/cfg/models/v10/yolov10nq2.yaml')
   

    # # # 檢查是否有已保存的權重，並載入
    # weights_path = 'C:/project/yolov10/runs/detect/train4/weights/last.pt'
    # if os.path.exists(weights_path):
    #     model.load(weights_path)  # 使用 model.load 載入已保存的權重
    #     print(f"成功載入權重: {weights_path}") 
    # else:
    #     print("未找到已保存的權重，從頭開始訓練。")

    # # 開始訓練
    model.train(
        data='C:/project/yolov10/data_cfg/dataset.yaml',  # 路徑到您的數據集 YAML 文件
        epochs=300,  # 訓練輪數         
        batch=64,  # 批次大小（可根據需求調整）
        imgsz=640,  # 圖片尺寸
        device=0,  # GPU 設備編號（如果有多個 GPU，可以用 '0,1' 等方式指定多個）
        save=True,  # 開啟保存功能
        save_period=20,  # 每隔多少個 epoch 保存一次權重
        plots =True,
        resume=True,
        patience=0
       
    )

    # # 訓練結束後進行模型驗證
    # model.val(
    #     data='C:/project/yolov10/data_cfg/dataset.yaml',  # 使用相同的數據集進行驗證
    #     batch=64,  # 驗證批次大小
    #     imgsz=640,  # 驗證時的圖像大小
    #     device=0  # 使用 GPU 驗證
    # )

    # # 訓練完成後保存最新的權重
    # weights_dir = 'C:/project/yolov10/latest_weights'
    # if not os.path.exists(weights_dir):
    #     os.makedirs(weights_dir)
    
    # torch.save(model.state_dict(), os.path.join(weights_dir, 'latest_model_state_dict.pt'))
    # print("訓練完成，權重已保存。")