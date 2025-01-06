import numpy as np
from scipy.optimize import minimize

class ImbalancedSampling:
    def __init__(self, r_values, z):
        """
        初始化
        r_values: 各类别样本比例列表（已按从大到小排序）
        z: 给定的约束参数
        """
        self.r_values = np.array(r_values)
        self.K = len(r_values)
        self.z = z
        
    def calculate_total_samples(self, s_A, s_B, i):
        """
        计算总样本量，确保A组和B组内部样本数量相等
        s_A: A组的采样比例
        s_B: B组的采样比例
        i: 分割点
        """
        if i == 0 or i == self.K:
            return 0
        
        r_A = self.r_values[:i]
        r_B = self.r_values[i:]
        
        samples_A = r_A * s_A
        N_A = min(samples_A) * i
        
        samples_B = r_B * s_B
        N_B = min(samples_B) * (self.K - i)
        
        return N_A + N_B
    
    def constraint_z(self, params, i):
        """修改为不等式约束：sqrt(N_B)/(N_A+N_B) >= z"""
        s_A, s_B = params[:i], params[i:]
        
        r_A = self.r_values[:i]
        r_B = self.r_values[i:]
        
        N_A = min(r_A * s_A) * i
        N_B = min(r_B * s_B) * (self.K - i)
        
        # 返回 z - sqrt(N_B)/(N_A+N_B) <= 0
        return z - np.sqrt(N_B) / (N_A + N_B + 1e-10)
    
    def constraints_equal_A(self, params, i):
        """确保A组内样本数量相等的约束"""
        s_A = params[:i]
        r_A = self.r_values[:i]
        samples_A = r_A * s_A
        
        constraints = []
        for j in range(i-1):
            constraints.append(samples_A[j] - samples_A[j+1])
        return constraints
    
    def constraints_equal_B(self, params, i):
        """确保B组内样本数量相等的约束"""
        s_B = params[i:]
        r_B = self.r_values[i:]
        samples_B = r_B * s_B
        
        constraints = []
        for j in range(len(r_B)-1):
            constraints.append(samples_B[j] - samples_B[j+1])
        return constraints
    
    def constraint_min_class(self, params, i):
        """确保B组最小类别的采样比例为1的约束"""
        s_B = params[i:]
        
        constraints = []
        
        if i < self.K:
            # B组最小类别（最后一个）采样比例应为1
            constraints.append(s_B[-1] - 1)
            
        return constraints
    
    def objective_function(self, params, i):
        """目标函数：最大化总样本量"""
        s_A, s_B = params[:i], params[i:]
        return -self.calculate_total_samples(s_A, s_B, i)
    
    def optimize(self):
        """
        对每个可能的i值进行优化，找到全局最优解
        """
        best_result = None
        best_value = float('-inf')
        best_i = None
        
        # 遍历所有可能的i值（至少要有1个类别在每组）
        for i in range(1, self.K):
            # 初始化采样比例，将最小类别的采样比例设为1
            initial_params = np.ones(self.K)
            
            # 定义约束条件
            constraints = [
            # 不等式约束
            {'type': 'ineq', 'fun': lambda x: -self.constraint_z(x, i)}
            ]
            
            # A组内样本数量相等约束
            for j in range(i-1):
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, j=j: self.constraints_equal_A(x, i)[j]
                })
            
            # B组内样本数量相等约束
            for j in range(self.K-i-1):
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, j=j: self.constraints_equal_B(x, i)[j]
                })
            
            # 最小类别采样比例为1的约束
            for j, cons in enumerate(self.constraint_min_class(initial_params, i)):
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, j=j: self.constraint_min_class(x, i)[j]
                })
            
            # 采样比例的边界
            bounds = [(0, 1) for _ in range(self.K)]
            
            # 优化
            try:
                result = minimize(
                    lambda x: self.objective_function(x, i),
                    initial_params,
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds,
                    options={'ftol': 1e-8, 'maxiter': 1000}
                )

                if result.success and -result.fun > best_value:
                    best_value = -result.fun
                    best_result = result.x
                    best_i = i
            except:
                continue
        
        return best_i, best_result, best_value

# 使用示例
if __name__ == "__main__":
    # 示例数据
    r_values = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]  # 示例类别比例
    z = 0.005  # 示例z值
    
    solver = ImbalancedSampling(r_values, z)
    best_i, best_s_values, max_total = solver.optimize()
    
    print(f"最优分割点 i: {best_i}")
    print(f"最优采样比例: {best_s_values}")
    print(f"最大总样本量: {max_total}")
    
    # 验证结果
    if best_i is not None:
        s_A = best_s_values[:best_i]
        s_B = best_s_values[best_i:]
        r_A = r_values[:best_i]
        r_B = r_values[best_i:]
        
        print("\n验证结果:")
        print(f"A组各类别采样比例: {s_A}")
        print(f"B组各类别采样比例: {s_B}")
        print(f"A组各类别采样后的相对数量: {r_A * s_A}")
        print(f"B组各类别采样后的相对数量: {r_B * s_B}")
        
        # 验证最小类别采样比例是否为1
        print(f"\nA组最小类别采样比例: {s_A[-1]}")
        print(f"B组最小类别采样比例: {s_B[-1]}")