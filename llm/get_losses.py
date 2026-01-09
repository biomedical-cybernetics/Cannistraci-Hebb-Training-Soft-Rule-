import os
import re
import numpy as np
def extract_floats_from_txt(folder_path):
    """
    从指定文件夹中的所有 txt 文件提取特定形式的浮点数，并返回文件名与数字的字典。
    :param folder_path: 文件夹路径
    :return: 字典，键为文件名，值为提取的浮点数
    """
    result = {}
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # 只处理 .txt 文件
            file_path = os.path.join(folder_path, file_name)
            if "s_0.7_" not in file_name:
                continue
            # 打开并读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 使用正则表达式匹配 xxxx 浮点数 yyy 的形式
                # Final eval loss: 3.5380586683750153
                match = re.search(r"Final eval loss: (\d+\.\d+)", content)
                # match = re.search(r"update_step: 10001, \{'final_eval_loss': (\d+\.\d+),", content)
                if match:
                    # 提取浮点数
                    float_number = np.exp(float(match.group(1)))
                    # 添加到结果字典
                    # file_name_stride = file_name[56:]
                    result[file_name] = float_number

    return result

# 示例：指定文件夹路径
folder_path = 'tmp/'  # 替换为你的文件夹路径
result = extract_floats_from_txt(folder_path)
for i in result.keys():
  print(i[:-4], ":  ", result[i])




# 打印结果
# print(result)