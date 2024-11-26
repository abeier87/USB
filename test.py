import numpy as np


def cosine_similarity_matrix(matrix):
    """
    计算矩阵中向量两两之间的夹角余弦值，返回K*K维的对称矩阵

    参数:
    matrix: 维度为(K, d)的numpy数组，表示K个d维向量

    返回:
    cos_matrix: K*K维的numpy数组，其元素(i, j)表示第i个向量和第j个向量的夹角余弦值
    """
    # 计算向量的模长，shape为(K, 1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 对矩阵进行归一化，使得每一行向量都变成单位向量
    normalized_matrix = matrix / norms
    # 通过矩阵乘法计算两两向量的点积，得到K*K的矩阵，其元素(i, j)表示第i个向量和第j个向量的点积
    dot_product_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    return dot_product_matrix

# 示例用法
# 随机生成一个维度为(5, 3)的矩阵，代表5个3维向量，你可以替换成自己真实的数据
K = 5
d = 3
matrix = np.random.rand(K, d)
result = cosine_similarity_matrix(matrix)
print(result)