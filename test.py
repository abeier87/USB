import numpy as np
import torch

# def cosine_similarity_matrix(matrix):
#     """
#     计算矩阵中向量两两之间的夹角余弦值，返回K*K维的对称矩阵

#     参数:
#     matrix: 维度为(K, d)的numpy数组，表示K个d维向量

#     返回:
#     cos_matrix: K*K维的numpy数组，其元素(i, j)表示第i个向量和第j个向量的夹角余弦值
#     """
#     # 计算向量的模长，shape为(K, 1)
#     norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#     # 对矩阵进行归一化，使得每一行向量都变成单位向量
#     normalized_matrix = matrix / norms
#     # 通过矩阵乘法计算两两向量的点积，得到K*K的矩阵，其元素(i, j)表示第i个向量和第j个向量的点积
#     dot_product_matrix = np.dot(normalized_matrix, normalized_matrix.T)
#     # 为了避免数值计算误差导致余弦值超出[-1, 1]范围，进行裁剪
#     clipped_dot_product_matrix = np.clip(dot_product_matrix, -1, 1)
#     # 通过反余弦函数（arccos）将点积（也就是余弦值）转换为角度值（单位为弧度）
#     angle_matrix = np.arccos(clipped_dot_product_matrix)
    
#     K = angle_matrix.shape[0]
#     # 创建数组用于存储每个向量的平均夹角值
#     average_angles = np.zeros(K)
#     for i in range(K):
#         # 排除与自身的夹角（值为0，因为向量与自身夹角为0弧度），计算其余夹角的平均值
#         average_angles[i] = np.mean(angle_matrix[i, np.arange(K)!= i])
    
#     total_sum = np.sum(average_angles)
#     proportions = average_angles / total_sum
#     return proportions


# 定义张量a
a = torch.tensor([[1, 2], [2, 2], [3, 1], [1, 0]])
# 定义张量b
b = torch.tensor([5, 5])

print(a+b)