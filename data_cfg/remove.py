import os
from tqdm import tqdm

def remove_labels(lable_path, label_type="6"):
    files = os.listdir(lable_path)
    conut = 0
    for file in tqdm(files, desc="Processing files", unit="file", colour="blue"):
        new_line = []

        if file.endswith(".txt"):
            file_path = os.path.join(lable_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if not line.startswith(label_type):
                    new_line.append(line)
                else:
                    conut +=1
            
            with open(file_path, "w") as f:
                f.writelines(new_line)
    print(f"remove {conut} lable ")

if __name__ == "__main__":
    lable_path = "C:/Users/user/Desktop/Ship Dataset/0923 _np_6/val/labels"
    remove_labels(lable_path)
    print("remove labels done")

# 431