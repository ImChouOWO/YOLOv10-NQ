import pickle

file_path = 'C:/project/yolov10/gate/gate_values.pkl'

# 讀取所有保存的 gate_value
gate_values = []
with open(file_path, 'rb') as f:
    while True:
        try:
            gate_values.append(pickle.load(f))
        except EOFError:
            break  # 文件結束

# 查看讀取到的所有 gate_values
print(len(gate_values))
