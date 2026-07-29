#!/usr/bin/env python3
"""
蛋白质投影LGCN去噪模块

该模块包含用于蛋白质投影矩阵去噪的LGCN相关功能：
1. 蛋白质投影矩阵的LGCN降噪处理
2. 线性GCN去噪处理
3. 基于三分图的元路径LightGCN处理
4. 相关的辅助函数

作者: HGCLDR项目
"""

import os
import numpy as np
import scipy.sparse as sp
from typing import Tuple


class ProteinDenoiser:
    """蛋白质投影去噪器"""
    
    def __init__(self, num_drugs: int, num_diseases: int, train_csr: sp.csr_matrix):
        """
        初始化蛋白质去噪器
        
        Args:
            num_drugs: 药物数量
            num_diseases: 疾病数量  
            train_csr: 训练观测矩阵 (D×Di)
        """
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        self.train_csr = train_csr
    
    def denoise_protein_projection_with_lgcn(self, R_proj: sp.csr_matrix) -> sp.csr_matrix:
        """
        专门用于蛋白质投影矩阵的LGCN降噪处理：
        1) 构建包含蛋白质投影的临时邻接矩阵
        2) 应用LGCN扩散机制
        3) 提取降噪后的蛋白质投影边
        4) 返回降噪后的U×I蛋白质投影矩阵
        
        环境变量：
          PROJ_LGCN_K     (默认 2)   扩散阶数
          PROJ_LGCN_TOPK  (默认 15)  每个药物保留的投影疾病数
          PROJ_LGCN_BETA  (默认 0.6) 投影边权重系数
        """
        D, Di = self.num_drugs, self.num_diseases
        K    = int(os.environ.get("PROJ_LGCN_K", "2"))
        TOPK = int(os.environ.get("PROJ_LGCN_TOPK", "15"))
        BETA = float(os.environ.get("PROJ_LGCN_BETA", "0.6"))
        
        # === LGCN++优化：K-support策略 ===
        MIN_SUP = int(os.environ.get("PROJ_MIN_SUP", "0"))
        MAX_SUP = int(os.environ.get("PROJ_MAX_SUP", "0"))
        SUP_METHOD = os.environ.get("PROJ_SUP_METHOD", "none")
        
        if SUP_METHOD == "k_support" and (MIN_SUP > 0 or MAX_SUP > 0):
            print(f"[PROJ-KSUPPORT] 应用K-support策略: MIN_SUP={MIN_SUP}, MAX_SUP={MAX_SUP}")
            # 这里需要A_dp和A_pd来计算支持数，但当前方法没有这些参数
            # 暂时跳过，在调用处处理
            pass
        
        # 1) 构建临时对称邻接矩阵用于LGCN处理
        # 只包含观测边和蛋白质投影边
        R_obs = self.train_csr.tocsr().astype(np.float32)
        R_temp = (R_obs + R_proj).tocsr()
        
        # 构建 (D+Di)×(D+Di) 对称方阵
        coo = R_temp.tocoo()
        rows = np.concatenate([coo.row, coo.col + D])
        cols = np.concatenate([coo.col + D, coo.row])
        data = np.concatenate([coo.data, coo.data]).astype(np.float32)
        A_temp = sp.coo_matrix((data, (rows, cols)), shape=(D + Di, D + Di)).tocsr()
        
        # 2) 对称归一化
        deg = np.asarray(A_temp.sum(1)).ravel().astype(np.float32)
        deg[deg == 0] = 1.0
        D_inv_sqrt = sp.diags((deg ** -0.5).astype(np.float32))
        N = (D_inv_sqrt @ A_temp @ D_inv_sqrt).tocsr()
        
        # 3) 扩散 M = N^K
        M = N.copy()
        for _ in range(1, K):
            M = (M @ N).tocsr()
        
        # 4) 提取D×Di部分并屏蔽观测边
        M_di = M[:D, D:D+Di].tocsr()
        M_di = M_di.tolil()
        ru, ci = R_obs.nonzero()
        M_di[ru, ci] = 0.0  # 屏蔽观测边
        M_di = M_di.tocsr()
        M_di.eliminate_zeros()
        
        # 5) Top-K筛选（只保留蛋白质投影相关的边）
        M_di_top = self._row_topk(M_di, TOPK)
        R_proj_denoised = (M_di_top * BETA).tocsr()
        R_proj_denoised.eliminate_zeros()
        
        print(f"[PROJ-LGCN] K={K}, topk={TOPK}, beta={BETA} | "
              f"R_proj_orig nnz={R_proj.nnz}, R_proj_denoised nnz={R_proj_denoised.nnz}")
        
        return R_proj_denoised

    def denoise_with_linear_gcn(self, adj_msg: sp.csr_matrix) -> sp.csr_matrix:
        """
        对 (D+Di)×(D+Di) 的消息传递邻接做一次"线性 GCN"去噪：
          1) N = D^{-1/2} A D^{-1/2}
          2) M = N^K
          3) 取 D×Di 块 M_di，先屏蔽观测位置，再做行 TopK 得到软边 R_soft
          4) 和原来的 D×Di 块 R_src_di 做并集（保留原投影信息），得到 R_merge
          5) 拼回对称方阵返回
        环境变量：
          PROJ_LGCN_K     (默认 3)   扩散阶数
          PROJ_LGCN_TOPK  (默认 36)  每个药物保留的疾病 TopK（去噪后新软边）
          PROJ_LGCN_BETA  (默认 0.30)软边权重系数
        """
        D, Di = self.num_drugs, self.num_diseases
        K    = int(os.environ.get("PROJ_LGCN_K", "3"))
        TOPK = int(os.environ.get("PROJ_LGCN_TOPK", "36"))
        BETA = float(os.environ.get("PROJ_LGCN_BETA", "0.30"))
    
        A = adj_msg.tocsr().astype(np.float32)
        R_obs = self.train_csr.tocsr().astype(np.float32)      # D×Di
        R_src_di = A[:D, D:D+Di].tocsr().astype(np.float32)     # 原消息传递的 D×Di（含投影）
    
        # 1) 对称归一化
        deg = np.asarray(A.sum(1)).ravel().astype(np.float32)
        deg[deg == 0] = 1.0
        D_inv_sqrt = sp.diags((deg ** -0.5).astype(np.float32))
        N = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()
    
        # 2) 扩散 M = N^K
        M = N.copy()
        for _ in range(1, K):
            M = (M @ N).tocsr()
    
        # 3) 取 D×Di，先屏蔽观测位置，再做 TopK
        M_di = M[:D, D:D+Di].tocsr()
        # 观测掩码：观测位置清零，确保 TopK 不会选到观测
        M_di = M_di.tolil()
        ru, ci = R_obs.nonzero()
        M_di[ru, ci] = 0.0
        M_di = M_di.tocsr(); M_di.eliminate_zeros()
    
        # 行 TopK（在非观测位置里选）
        M_di_top = self._row_topk(M_di, TOPK)
        R_soft = (M_di_top * BETA).tocsr()
        R_soft.eliminate_zeros()
    
        # 4) 和原 D×Di 做并集（保留原投影边 + 新软边）
        R_merge = (R_src_di + R_soft).tocsr()
        R_merge.eliminate_zeros()
    
        # 5) 拼回 (D+Di)×(D+Di) 对称方阵
        coo = R_merge.tocoo()
        rows = np.concatenate([coo.row, coo.col + D])
        cols = np.concatenate([coo.col + D, coo.row])
        data = np.concatenate([coo.data, coo.data]).astype(np.float32)
        A_clean = sp.coo_matrix((data, (rows, cols)), shape=(D + Di, D + Di)).tocsr()
    
        print(f"[DENOISE-LGCN] K={K}, topk={TOPK}, beta={BETA} | "
              f"R_src_di nnz={R_src_di.nnz}, R_soft nnz={R_soft.nnz}, "
              f"A_clean nnz={A_clean.nnz}")
        return A_clean

    def _row_topk(self, M: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """对 CSR 做每行 Top-K 筛选（按数值），k<=0 原样返回。"""
        if k <= 0:
            return M
        M = M.tocsr().astype(np.float32)
        indptr, indices, data = M.indptr, M.indices, M.data
        rows, cols, vals = [], [], []
        for i in range(M.shape[0]):
            s, e = indptr[i], indptr[i+1]
            if e - s <= k:
                rows.extend([i] * (e - s))
                cols.extend(indices[s:e])
                vals.extend(data[s:e])
            else:
                idx = np.argpartition(data[s:e], -k)[-k:] + s
                idx = idx[np.argsort(-data[idx])]
                rows.extend([i] * len(idx))
                cols.extend(indices[idx])
                vals.extend(data[idx])
        return sp.csr_matrix((np.asarray(vals, np.float32),
                            (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
                            shape=M.shape)


def create_protein_denoiser(num_drugs: int, num_diseases: int, train_csr: sp.csr_matrix) -> ProteinDenoiser:
    """
    创建蛋白质去噪器的工厂函数
    
    Args:
        num_drugs: 药物数量
        num_diseases: 疾病数量
        train_csr: 训练观测矩阵
        
    Returns:
        ProteinDenoiser实例
    """
    return ProteinDenoiser(num_drugs, num_diseases, train_csr)


# 为了向后兼容，提供独立的函数接口
def denoise_protein_projection_with_lgcn(R_proj: sp.csr_matrix, 
                                        num_drugs: int, 
                                        num_diseases: int, 
                                        train_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    独立的蛋白质投影LGCN去噪函数（向后兼容）
    
    Args:
        R_proj: 蛋白质投影矩阵 (D×Di)
        num_drugs: 药物数量
        num_diseases: 疾病数量
        train_csr: 训练观测矩阵 (D×Di)
        
    Returns:
        去噪后的蛋白质投影矩阵 (D×Di)
    """
    denoiser = create_protein_denoiser(num_drugs, num_diseases, train_csr)
    return denoiser.denoise_protein_projection_with_lgcn(R_proj)


def denoise_with_linear_gcn(adj_msg: sp.csr_matrix,
                           num_drugs: int,
                           num_diseases: int, 
                           train_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    独立的线性GCN去噪函数（向后兼容）
    
    Args:
        adj_msg: 消息传递邻接矩阵 (D+Di)×(D+Di)
        num_drugs: 药物数量
        num_diseases: 疾病数量
        train_csr: 训练观测矩阵 (D×Di)
        
    Returns:
        去噪后的邻接矩阵 (D+Di)×(D+Di)
    """
    denoiser = create_protein_denoiser(num_drugs, num_diseases, train_csr)
    return denoiser.denoise_with_linear_gcn(adj_msg)


