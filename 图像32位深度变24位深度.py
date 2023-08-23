from PIL import Image
import os

folder_path = "path/to/folder"  # 文件夹路径

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

for file_name in file_names:
    if file_name.endswith(".jpg"):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 打开图像
        image = Image.open(file_path)
        # 将图像转换为24位深度
        image = image.convert("RGB")
        
        # 保存修改后的图像
        image.save(file_path)
        print(f"已将 {file_name} 的位深度从32改为24")
