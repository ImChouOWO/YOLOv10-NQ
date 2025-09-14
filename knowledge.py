# import sys
# import os
# from ultralytics import YOLOv10 #student
# # 動態加入教師模型專案的路徑
# teacher_project_path = "C:/project/origin_yolov10"  # 替換為教師模型專案的路徑
# if teacher_project_path not in sys.path:
#     sys.path.append(teacher_project_path)

# # 導入教師模型的 YOLOv10
# from ultralytics import YOLOv10 as TeacherYOLOv10  

# if __name__ == '__main__':

#     model = TeacherYOLOv10('C:/project/origin_yolov10/yolov10/ultralytics/cfg/models/v10/yolov10b.yaml')

#     # 檢查是否有已保存的權重，並載入
#     weights_path = 'C:/project/origin_yolov10/run/test/train3/weights/best.pt'
#     if os.path.exists(weights_path):
#         model.load(weights_path)  # 使用 model.load 載入已保存的權重
#         print(f"成功載入權重: {weights_path}")
#     else:
#         print("未找到已保存的權重，從頭開始訓練。")
#     model.val(
#         data='C:/project/yolov10/data_cfg/dataset.yaml',  # 使用相同的數據集進行驗證
#         batch=32,  # 驗證批次大小
#         imgsz=640,  # 驗證時的圖像大小
#         device=0  # 使用 GPU 驗證
#     )
import argparse
from ultralytics import YOLOv10  # 使用 YOLOv10

# 定義 parser 並添加命令列參數
parser = argparse.ArgumentParser(description="YOLOv10 Training with Teacher Model Support")

# 必要參數
parser.add_argument('--data', type=str, required=True, help='Path to dataset configuration file (e.g., coco.yaml)')
parser.add_argument('--cfg', type=str, required=True, help='Path to student model configuration file (e.g., yolov10nq2.yaml)')
parser.add_argument('--weights', type=str, default='', help='Path to student model weights (if any)')

# 教師模型參數
parser.add_argument('--teacher_weight', type=str, default='', help='Path to teacher model weights (e.g., yolov10l.pt)')
parser.add_argument('--teacher_cfg', type=str, default='', help='Path to teacher model configuration file (e.g., yolov10l.yaml)')

# 訓練參數
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--device', type=str, default='0', help='Device for training (e.g., 0, 0,1 for multiple GPUs)')

# 解析命令列參數
opt = parser.parse_args()

# 打印參數以便檢查
print("Training Parameters:")
print(opt)

