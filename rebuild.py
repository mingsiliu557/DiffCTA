import os
import pandas as pd

# 图像文件夹路径
image_dir = '/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/data/SYSU/image'

# 输出CSV文件路径
output_csv_path = '/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/data/SYSU.csv'

# 创建空列表来保存数据
image_mask_data = []

# 遍历图像文件夹中的所有文件
for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)
    
    # 检查是否是图片文件（jpeg格式）
    if os.path.isfile(image_path) and image_filename.endswith('.jpg'):
        # 相对路径，保留目录结构
        relative_image_path = os.path.join('SYSU', 'image', image_filename)
        
        # mask路径（与图像路径相同）
        mask_path = relative_image_path
        
        # 将数据加入列表
        image_mask_data.append([relative_image_path, mask_path])

# 将列表转换为DataFrame
df_image_mask = pd.DataFrame(image_mask_data, columns=['image', 'mask'])

# 将DataFrame保存为CSV文件
df_image_mask.to_csv(output_csv_path, index=False)

print(f'CSV文件已保存到 {output_csv_path}')
