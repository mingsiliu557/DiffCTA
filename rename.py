import os
import pandas as pd

# 文件夹路径
image_dir = '/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/data/SYSU/images/'

# 标签文件路径
reclassified_file_path = '/data1/zhangxingshuai/lms/OPTIC_CLASSIFY/data/SYSU/c5_DR_reclassified.csv'

# 读取重新分类的标签文件
reclassified_df = pd.read_csv(reclassified_file_path)

# 创建一个函数来重新分类并重命名文件
def rename_reclassified_images(reclassified_df):
    for _, row in reclassified_df.iterrows():
        image_name = row['Fundus_images']  # 获取图像文件名
        new_dr_grade = row['DR_grade(International_Clinical_DR_Severity_Scale)']  # 获取新分类标签

        # 构建旧的和新的文件路径
        old_image_name = f'5_{image_name}'  # 原文件名为 5_开头
        old_image_path = os.path.join(image_dir, old_image_name)
        new_image_name = f'{new_dr_grade}_{image_name}'  # 使用新的标签作为前缀
        new_image_path = os.path.join(image_dir, new_image_name)

        # 重命名文件
        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)
            print(f'Renamed {old_image_path} to {new_image_path}')
        else:
            print(f'File not found: {old_image_path}')

# 执行重新分类并重命名操作
rename_reclassified_images(reclassified_df)

print("重新分类并重命名完成！")
