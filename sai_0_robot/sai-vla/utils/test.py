import numpy as np
import math


def polyfit_chunk(frames: np.ndarray, values: np.ndarray, degree: int = 3):
    """
    给定数据点 (x_i, y_i)，找 n 阶多项式 p(x) = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
    使得 Σ(y_i - p(x_i))^2 最小
    
    构建范德蒙矩阵 V：
        V[i,j] = x_i^j, 其中 i=0..m-1, j=0..n
    
    解正规方程：(V^T * V) * a = V^T * y
    系数向量：a = (V^T * V)^(-1) * V^T * y
    """
    x = frames.astype(np.float64)
    y = values.astype(np.float64)
    n = len(x)
    
    # 数值稳定性处理：对 x 进行归一化
    x_mean = np.mean(x)
    x_std = np.std(x) if np.std(x) > 0 else 1.0
    x_norm = (x - x_mean) / x_std
    
    # 构建范德蒙矩阵 V[i,j] = x_norm[i]^j
    # V 的形状是 (n, degree+1)
    V = np.zeros((n, degree + 1))
    for j in range(degree + 1):
        V[:, j] = x_norm ** j
    
    # 解正规方程：(V^T * V) * a = V^T * y
    # a = (V^T * V)^(-1) * V^T * y
    VtV = V.T @ V  # (degree+1, degree+1)
    Vty = V.T @ y  # (degree+1,)
    
    # 使用求解线性方程组（比直接求逆更稳定）
    coeffs_norm = np.linalg.solve(VtV, Vty)  # [a_0, a_1, ..., a_n]
    
    # 计算拟合值
    fitted_values = V @ coeffs_norm
    
    # 将归一化系数转换回原始坐标系的系数
    # p(x) = Σ a_j * ((x - x_mean) / x_std)^j
    # 展开后得到原始 x 的系数
    coeffs_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        # 二项式展开: ((x - x_mean) / x_std)^j = Σ C(j,k) * (-x_mean/x_std)^(j-k) * (x/x_std)^k
        for k in range(j + 1):
            binom_coeff = math.comb(j, k)
            term = binom_coeff * ((-x_mean / x_std) ** (j - k)) * (1 / x_std) ** k
            coeffs_original[k] += coeffs_norm[j] * term
    
    # 转换为高次到低次的顺序 [a_n, a_{n-1}, ..., a_0]
    coeffs = coeffs_original[::-1]
    
    # R²
    ss_res = np.sum((y - fitted_values) ** 2) 
    ss_tot = np.sum((y - np.mean(y)) ** 2)      
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return fitted_values, coeffs, r_squared