import os
import pickle as pkl
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from reckit.random import randint_choice
from utils.util import sparse_mx_to_torch_sparse_tensor, normalize
from utils.protein_denoise import denoise_protein_projection_with_lgcn, denoise_with_linear_gcn, ProteinDenoiser



class Data(object):
    def __init__(
        self, dataset, norm_adj, seed, ssl_ratio=0.1,
        fold=None, split_variant='selection'
    ):
        pkl_path = os.path.join('./data/' + dataset)
        self.pkl_path = pkl_path
        self.dataset = dataset
        self.seed = int(seed)
        self.ssl_ratio = ssl_ratio
        if fold is None:
            raise ValueError('This release requires fold=0..9')
        self.fold = int(fold)
        self.split_variant = split_variant
        fold = self.fold
        if not 0 <= fold <= 9:
            raise ValueError(f'fold must be between 0 and 9, got {fold}')
        fold_path = os.path.join(pkl_path, 'folds', f'fold_{fold:02d}')
        if split_variant == 'selection':
            self.split_path = fold_path
        elif split_variant == 'refit':
            self.split_path = os.path.join(fold_path, 'refit')
        else:
            raise ValueError(f'Unknown split_variant: {split_variant}')
        required = ['train.pkl', 'val.pkl', 'test.pkl', 'adj_csr.npz', 'manifest.json']
        missing = [name for name in required if not os.path.exists(os.path.join(self.split_path, name))]
        if missing:
            raise FileNotFoundError(f'Missing fold files in {self.split_path}: {missing}')
        self.drug_disease_list = self.load_pickle(os.path.join(pkl_path, 'drug_disease_list.pkl'))
    
        # 1) 过滤空药物并重映射
        filtered = []
        old2new = {}
        for old_d, diseases in enumerate(self.drug_disease_list):
            if diseases:
                old2new[old_d] = len(filtered)
                filtered.append(sorted(set(diseases)))
        if not filtered:
            raise ValueError(f"[{dataset}] All drugs have empty interactions. "
                             f"Please check data/{dataset}/drug_disease_list.pkl")
        self.drug_disease_list = filtered
        self.num_drugs = len(self.drug_disease_list)
        self.num_diseases = max(max(diseases) for diseases in self.drug_disease_list) + 1
    
        # 2) 划分 train/test
        train_path = os.path.join(self.split_path, 'train.pkl')
        val_path = os.path.join(self.split_path, 'val.pkl')
        test_path = os.path.join(self.split_path, 'test.pkl')
        if os.path.exists(train_path):
            self.train_dict = self.load_pickle(train_path)
            self.test_dict = self.load_pickle(test_path)
            self.val_dict = (
                self.load_pickle(val_path)
                if os.path.exists(val_path)
                else {drug: [] for drug in range(self.num_drugs)}
            )
        else:
            raise FileNotFoundError(f'Cross-validation split not found: {self.split_path}')
        print(
            f'[SPLIT] dataset={dataset}, fold={fold}, variant={split_variant}, '
            f'path={self.split_path}'
        )
    
        # 3) 训练对列表 & 观测邻接（采样/监督用）
        train_list = []
        for d, diseases in self.train_dict.items():
            train_list.extend([[d, di] for di in diseases])
        self.train_np = np.array(train_list, dtype=np.int32)
        self.adj_train, _ = self.generate_adj()
    
        print('num_drugs %d, num_diseases %d' % (self.num_drugs, self.num_diseases))
        print('adjacency (for sampler) shape: ', self.adj_train.shape)
        tot_num_rating = sum([len(x) for x in self.drug_disease_list])
        print('number of all ratings {}, density {:.6f}'.format(
            tot_num_rating, tot_num_rating / (self.num_drugs * self.num_diseases)
        ))
    
        # 4) 训练/测试 CSR （D×Di）
        self.train_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_drugs, self.num_diseases)
        self.val_csr = self.generate_rating_matrix([*self.val_dict.values()], self.num_drugs, self.num_diseases)
        self.test_csr  = self.generate_rating_matrix([*self.test_dict.values()],  self.num_drugs, self.num_diseases)
    
        # 5) 投影控制项 & 路径 & P 数
        self.proj_topk   = int(os.environ.get("PROJ_TOPK", 40))
        self.proj_weight = float(os.environ.get("PROJ_WEIGHT", 1.0))
        default_dp_path = os.path.join(pkl_path, 'DrugProteinAssociationNumber.csv')
        default_pd_path = os.path.join(pkl_path, 'ProteinDiseaseAssociationNumber.csv')
        self.dp_path_overridden = 'DP_EDGES' in os.environ
        self.pd_path_overridden = 'PD_EDGES' in os.environ
        self.dp_path = os.environ.get('DP_EDGES', default_dp_path)
        self.pd_path = os.environ.get('PD_EDGES', default_pd_path)
        missing_projection_files = [
            path for path in (self.dp_path, self.pd_path) if not os.path.isfile(path)
        ]
        if missing_projection_files:
            raise FileNotFoundError(
                f'[{dataset}] Missing protein projection files: {missing_projection_files}'
            )
        self.num_proteins = int(os.environ.get("NUM_PROTEINS", -1))
        if self.num_proteins <= 0:
            self.num_proteins = self._infer_num_proteins(self.dp_path, self.pd_path)
        print(f"[INFO] num_proteins={self.num_proteins} | dp={self.dp_path} | pd={self.pd_path}")
    
        # 6) 构建消息传递邻接（观测 ∪ 二跳TopK）
        self.adj_train_msg = self._build_message_passing_adj()   # (D+Di)×(D+Di) CSR
        print('adjacency (for message passing) shape: ', self.adj_train_msg.shape)
        print(f"[INFO] nnz sampler={self.adj_train.count_nonzero()}, "
              f"nnz msg={self.adj_train_msg.count_nonzero()}, "
              f"added={self.adj_train_msg.count_nonzero() - self.adj_train.count_nonzero()}")
    
        # 6.1) 线性GCN去噪（已禁用，改用专门的蛋白质投影LGCN降噪）
        # if int(os.environ.get("DENOISE_LGCN", "0")):  # 默认禁用
        #     self.adj_train_msg = self._denoise_with_linear_gcn(self.adj_train_msg)
    
        # 7) 归一化+转稀疏张量（编码用）
        if norm_adj:
            n_nodes = self.adj_train_msg.shape[0]
            eye = sp.eye(n_nodes, dtype=np.float32, format='csr')
            self.adj_train_norm = normalize(self.adj_train_msg + eye)
            self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)
    
        # 8) 同构相似图 & ban/困难池
        self.load_similarity_graphs()
        self._build_ban_and_neg_pool(pool_topk=50)



    def generate_adj(self):
        """
        构建二分图邻接矩阵（药物+疾病，共 n_nodes = D+Di 个节点），返回：
          - adj_csr: 形状严格为 (n_nodes, n_nodes) 的 CSR 稀疏矩阵
          - drug_disease: 形状 (D, Di) 的 0/1 numpy 数组
        """
        # 1) 构 0/1 交互矩阵（D×Di）
        drug_disease = np.zeros((self.num_drugs, self.num_diseases), dtype=np.int32)
        for d, diseases in self.train_dict.items():
            if diseases:
                drug_disease[d, diseases] = 1
    
        n_nodes = self.num_drugs + self.num_diseases
        cache_path = os.path.join(self.split_path, 'adj_csr.npz')
    
        # 2) 先尝试加载缓存，但要做严格形状校验；不符合就重建
        if os.path.exists(cache_path):
            adj_csr = sp.load_npz(cache_path)
            if adj_csr.shape != (n_nodes, n_nodes):
                # 形状不匹配（很可能是旧缓存），重建并覆盖
                adj_csr = None
        else:
            adj_csr = None
    
        if adj_csr is None:
            # 3) 用显式 shape 构造严格的 n×n 方阵
            coo_drug_disease = sp.coo_matrix(drug_disease)
            # 构造双向边（药物→疾病、疾病→药物），索引落在 [0, n_nodes)
            rows = np.concatenate([coo_drug_disease.row, coo_drug_disease.col + self.num_drugs])
            cols = np.concatenate([coo_drug_disease.col + self.num_drugs, coo_drug_disease.row])
            data = np.ones(rows.shape[0], dtype=np.float32)
            # 关键：显式指定 shape=(n_nodes, n_nodes)
            adj_csr = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
            sp.save_npz(cache_path, adj_csr)
    
        return adj_csr, drug_disease

    def _infer_num_proteins(self, dp_path, pd_path):
        """从两份边文件里推断 NUM_PROTEINS = max(protein_id)+1"""
        def find_max(path, protein_col_candidates=('protein', 1)):
            if not os.path.exists(path):
                return -1
            try:
                df = pd.read_csv(path)
                if 'protein' in df.columns:
                    return int(df['protein'].max())
            except Exception:
                pass
            # 退化为无表头两列
            df = pd.read_csv(path, header=None)
            return int(df.iloc[:, 1].max())  # 第二列当 protein
        m1 = find_max(dp_path)
        m2 = find_max(pd_path)
        return max(m1, m2) + 1

    def _load_bipartite_edges_named(self, path, n_rows, n_cols, row_colname, col_colname):
        """
        读取带表头/无表头的二部边文件，返回 n_rows×n_cols 的 0/1 CSR。
        - row_colname: 行索引使用的列名（不存在时默认第1列）
        - col_colname: 列索引使用的列名（不存在时默认第2列）
        """
        if not (path and os.path.exists(path)):
            return None
        try:
            df = pd.read_csv(path, sep=None, engine='python')
            if row_colname in df.columns and col_colname in df.columns:
                r = df[row_colname].astype(int).to_numpy()
                c = df[col_colname].astype(int).to_numpy()
            else:
                # 无表头/列名不匹配 → 按前两列读取
                df = pd.read_csv(path, header=None, sep=None, engine='python')
                r = df.iloc[:, 0].astype(int).to_numpy()
                c = df.iloc[:, 1].astype(int).to_numpy()
        except Exception:
            df = pd.read_csv(path, header=None, sep=None, engine='python')
            r = df.iloc[:, 0].astype(int).to_numpy()
            c = df.iloc[:, 1].astype(int).to_numpy()

        mask = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
        r, c = r[mask], c[mask]
        data = np.ones_like(r, dtype=np.float32)
        return sp.csr_matrix((data, (r, c)), shape=(n_rows, n_cols))
        
    def _load_pd_edges(self, path, n_proteins, n_diseases):
        """
        读取 Protein–Disease 边：
        - 有表头: 列名 'protein','disease'（注意顺序）
        - 无表头: 默认文件为 'disease,protein' 或 'disease protein'，需要交换列 -> (row=protein, col=disease)
        返回: P×Di 的 0/1 CSR
        """
        if not (path and os.path.exists(path)):
            return None
        try:
            df = pd.read_csv(path, sep=None, engine='python')
            if 'protein' in df.columns and 'disease' in df.columns:
                r = df['protein'].astype(int).to_numpy()
                c = df['disease'].astype(int).to_numpy()
            else:
                # 无表头/列名不匹配：按前两列读取，并交换
                df = pd.read_csv(path, header=None, sep=None, engine='python')
                # 文件列序: 0=disease, 1=protein  -> 需要 row=protein, col=disease
                r = df.iloc[:, 1].astype(int).to_numpy()  # protein
                c = df.iloc[:, 0].astype(int).to_numpy()  # disease
        except Exception:
            df = pd.read_csv(path, header=None, sep=None, engine='python')
            r = df.iloc[:, 1].astype(int).to_numpy()
            c = df.iloc[:, 0].astype(int).to_numpy()
    
        mask = (r >= 0) & (r < n_proteins) & (c >= 0) & (c < n_diseases)
        r, c = r[mask], c[mask]
        data = np.ones_like(r, dtype=np.float32)
        return sp.csr_matrix((data, (r, c)), shape=(n_proteins, n_diseases))

    def _gip_path(self, entity):
        filename = 'DrugGIP' if entity == 'drug' else 'DiseaseGIP'
        return os.path.join(self.split_path, filename + '.npy')

    def _read_similarity_matrix(self, path, expected_dim):
        if path.lower().endswith('.npy'):
            matrix = np.load(path, allow_pickle=False).astype(np.float32)
        else:
            matrix = pd.read_csv(path, index_col=0).values.astype(np.float32)
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f'Matrix shape mismatch in {path}: {matrix.shape}, '
                f'expected {(expected_dim, expected_dim)}'
            )
        return matrix

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
    
    def _similarity_augment(self, R_obs: sp.csr_matrix) -> sp.csr_matrix:
        """
        在蛋白质投影之前，先用相似度矩阵对观测矩阵进行结构增强
        参数：
          - R_obs: D×Di 的观测矩阵
        返回：
          - R_aug: 增强后的 D×Di 矩阵
        环境变量：
          - SIM_AUG_ENABLE: 是否启用相似度增强（默认0）
          - SIM_AUG_TOPK: 每行保留的TopK相似边（默认40）
          - SIM_AUG_WEIGHT: 相似度增强边的权重系数（默认0.3）
          - SIM_AUG_THRESHOLD: 相似度阈值，低于此值的边会被过滤（默认0.0）
        """
        if not int(os.environ.get("SIM_AUG_ENABLE", "0")):
            return R_obs
        
        D, Di = self.num_drugs, self.num_diseases
        topk = int(os.environ.get("SIM_AUG_TOPK", "40"))
        weight = float(os.environ.get("SIM_AUG_WEIGHT", "0.3"))
        threshold = float(os.environ.get("SIM_AUG_THRESHOLD", "0.0"))
        
        print(f"[SIM_AUG] 启用相似度增强: topk={topk}, weight={weight}, threshold={threshold}")
        
        # 加载原始相似度矩阵（CSV格式，需要重新读取）
        def load_sim_matrix(path, dim):
            if not os.path.exists(path):
                return None
            mat = self._read_similarity_matrix(path, dim)
            # 对称化
            mat = np.maximum(mat, mat.T)
            # 去对角线
            np.fill_diagonal(mat, 0.0)
            # 阈值过滤
            if threshold > 0:
                mat[mat < threshold] = 0.0
            return sp.csr_matrix(mat)
        
        # 加载四个相似度矩阵
        drug_gip = load_sim_matrix(self._gip_path('drug'), D)
        drug_fp = load_sim_matrix(os.path.join(self.pkl_path, 'DrugFingerprint.csv'), D)
        disease_gip = load_sim_matrix(self._gip_path('disease'), Di)
        disease_ps = load_sim_matrix(os.path.join(self.pkl_path, 'DiseasePS.csv'), Di)
        
        # 融合同类相似度矩阵（平均）
        drug_sim = None
        if drug_gip is not None and drug_fp is not None:
            drug_sim = (drug_gip + drug_fp) / 2.0
        elif drug_gip is not None:
            drug_sim = drug_gip
        elif drug_fp is not None:
            drug_sim = drug_fp
            
        disease_sim = None
        if disease_gip is not None and disease_ps is not None:
            disease_sim = (disease_gip + disease_ps) / 2.0
        elif disease_gip is not None:
            disease_sim = disease_gip
        elif disease_ps is not None:
            disease_sim = disease_ps
        
        # TopK筛选
        if topk > 0:
            if drug_sim is not None:
                drug_sim = self._row_topk(drug_sim, topk)
            if disease_sim is not None:
                disease_sim = self._row_topk(disease_sim, topk)
        
        # 药物侧增强: 相似药物的疾病关联
        R_drug_aug = sp.csr_matrix((D, Di), dtype=np.float32)
        if drug_sim is not None:
            # drug_sim @ R_obs: 对于每个药物，聚合相似药物的疾病关联
            R_drug_aug = (drug_sim @ R_obs).tocsr()
            print(f"[SIM_AUG] 药物侧增强: nnz={R_drug_aug.nnz}")
        
        # 疾病侧增强: 相似疾病的药物关联
        R_disease_aug = sp.csr_matrix((D, Di), dtype=np.float32)
        if disease_sim is not None:
            # R_obs @ disease_sim.T: 对于每个疾病，聚合相似疾病的药物关联
            R_disease_aug = (R_obs @ disease_sim.T).tocsr()
            print(f"[SIM_AUG] 疾病侧增强: nnz={R_disease_aug.nnz}")
        
        # 合并增强
        R_sim_total = (R_drug_aug + R_disease_aug).tocsr()
        
        # 加权合并到观测矩阵
        R_aug = (R_obs + weight * R_sim_total).tocsr()
        
        print(f"[SIM_AUG] 增强前 nnz={R_obs.nnz}, 增强后 nnz={R_aug.nnz}, 新增边={R_aug.nnz - R_obs.nnz}")
        
        return R_aug

    def _build_message_passing_adj(self) -> sp.csr_matrix:
        """
        用 观测 D–Di（train_csr） + 二跳投影 D–Di（A_dp @ A_pd） 构建消息传递用的方阵邻接。
        新增：
          - SIM_AUG_*: 相似度增强参数（在蛋白质投影之前）
          - PROJ_MIN_SUP：最小支持数（≥k 个 protein 共同支持才保留该二跳边）
          - PROJ_ALPHA  ：蛋白度惩罚系数（对高频 protein 降权，A_dp * Dp^{-α} * A_pd）
          - PROJ_TOPK   ：每个药物保留的投影疾病个数（Top-K）
          - PROJ_WEIGHT ：投影边在消息传递中的权重
        返回：(D+Di)×(D+Di) 的 CSR 方阵（对称，保留权重）
        """
        D, Di = self.num_drugs, self.num_diseases
    
        # 1) 观测 D–Di（训练划分）
        R_obs_raw = self.train_csr.tocsr().astype(np.float32)  # D×Di
        
        # 1.5) 【新增】相似度增强（在蛋白质投影之前）
        R_obs = self._similarity_augment(R_obs_raw)
    
        # 2) 载入 D–P、P–Di
        P = int(self.num_proteins)
        A_dp = self._load_bipartite_edges_named(
            self.dp_path, D, P, row_colname='drug', col_colname='protein'
        )
        # ProteinDiseaseAssociationNumber.csv 的表头是 "disease,protein"，用专门的读取函数读成 P×Di
        A_pd = self._load_pd_edges(self.pd_path, P, Di)
    
        print(f"[PROJ] A_dp shape={A_dp.shape if A_dp is not None else None}, nnz={A_dp.nnz if A_dp is not None else 0}")
        print(f"[PROJ] A_pd shape={A_pd.shape if A_pd is not None else None}, nnz={A_pd.nnz if A_pd is not None else 0}")
    
        if A_dp is None or A_pd is None:
            # 没有投影数据：仅用观测图
            coo = R_obs.tocoo()
            rows = np.concatenate([coo.row, coo.col + D])
            cols = np.concatenate([coo.col + D, coo.row])
            data = np.ones_like(rows, dtype=np.float32)
            return sp.coo_matrix((data, (rows, cols)), shape=(D + Di, D + Di)).tocsr()
    
        # === 使用LGCN投影方法 ===
        def row_norm(X: sp.csr_matrix) -> sp.csr_matrix:
            X = X.tocsr().astype(np.float32)
            rs = np.asarray(X.sum(1)).flatten()
            rs[rs == 0] = 1.0
            return sp.diags(1.0 / rs) @ X

        # === LGCN++优化：自适应度数抑制 + 双向归一化 ===
        ALPHA_METHOD = os.environ.get("PROJ_ALPHA_METHOD", "global")
        
        if ALPHA_METHOD == "adaptive":
            # 自适应度数抑制：按蛋白度数分位数给不同α值
            degP = np.asarray(A_dp.sum(0)).ravel().astype(np.float32) + 1e-8
            q90 = np.percentile(degP, 90)
            q50 = np.percentile(degP, 50)
            q10 = np.percentile(degP, 10)
            
            # 按分位数分配α值
            alpha_q90 = float(os.environ.get("PROJ_ALPHA_Q90", "0.8"))
            alpha_q50 = float(os.environ.get("PROJ_ALPHA_Q50", "0.5"))
            alpha_q10 = float(os.environ.get("PROJ_ALPHA_Q10", "0.2"))
            
            alpha_per_protein = np.where(degP >= q90, alpha_q90,
                                       np.where(degP >= q50, alpha_q50, alpha_q10))
            
            Dp_inv = sp.diags(np.power(degP, -alpha_per_protein))
            print(f"[PROJ-ADAPTIVE] 蛋白度数分位数: Q10={q10:.1f}, Q50={q50:.1f}, Q90={q90:.1f}")
            print(f"[PROJ-ADAPTIVE] α值分配: Q90+={alpha_q90}, Q50-Q90={alpha_q50}, Q10-={alpha_q10}")
            
        else:
            # 传统全局α值
            ALPHA = float(os.environ.get("PROJ_ALPHA", os.environ.get("ALPHA", "1.0")))
            if ALPHA > 0:
                degP = np.asarray(A_dp.sum(0)).ravel().astype(np.float32) + 1e-8
                Dp_inv = sp.diags(np.power(degP, -ALPHA))
            else:
                Dp_inv = sp.eye(A_dp.shape[1])
        
        # 双向归一化：RowNorm(A_dp) · Dp_inv · ColNorm(A_pd)
        NORM_METHOD = os.environ.get("PROJ_NORM_METHOD", "row")
        if NORM_METHOD == "bidirectional":
            # 双向归一化
            def col_norm(X: sp.csr_matrix) -> sp.csr_matrix:
                X = X.tocsr().astype(np.float32)
                cs = np.asarray(X.sum(0)).flatten()
                cs[cs == 0] = 1.0
                return X @ sp.diags(1.0 / cs)
            
            R_proj_raw = (row_norm(A_dp) @ Dp_inv @ col_norm(A_pd)).tocsr()
            print(f"[PROJ-BIDIR] 使用双向归一化")
        else:
            # 传统行归一化
            R_proj_raw = (row_norm(A_dp) @ Dp_inv @ row_norm(A_pd)).tocsr()
            print(f"[PROJ-ROW] 使用传统行归一化")
        
        print(f"[PROJ] R_proj_raw (before LGCN) nnz={R_proj_raw.nnz}")
        
        # === LGCN++优化：K-support策略 ===
        MIN_SUP = int(os.environ.get("PROJ_MIN_SUP", "0"))
        MAX_SUP = int(os.environ.get("PROJ_MAX_SUP", "0"))
        SUP_METHOD = os.environ.get("PROJ_SUP_METHOD", "none")
        
        if SUP_METHOD == "k_support" and (MIN_SUP > 0 or MAX_SUP > 0):
            print(f"[PROJ-KSUPPORT] 应用K-support策略: MIN_SUP={MIN_SUP}, MAX_SUP={MAX_SUP}")
            
            # 计算每个(u,i)对的蛋白支持数
            S_cnt = (A_dp.astype(bool).astype(np.int8) @ A_pd.astype(bool).astype(np.int8))  # D×Di
            
            # 应用最小支持筛选
            if MIN_SUP > 0:
                min_mask = (S_cnt >= MIN_SUP).astype(np.int8)
                R_proj_raw = R_proj_raw.multiply(min_mask).tocsr()
                R_proj_raw.eliminate_zeros()
                print(f"[PROJ-KSUPPORT] 最小支持筛选后: nnz={R_proj_raw.nnz}")
            
            # 应用最大支持筛选
            if MAX_SUP > 0:
                max_mask = (S_cnt <= MAX_SUP).astype(np.int8)
                R_proj_raw = R_proj_raw.multiply(max_mask).tocsr()
                R_proj_raw.eliminate_zeros()
                print(f"[PROJ-KSUPPORT] 最大支持筛选后: nnz={R_proj_raw.nnz}")
        
        # === LGCN++优化：残差门控 ===
        GATE_LAMBDA = float(os.environ.get("PROJ_GATE_LAMBDA", "0.5"))
        if GATE_LAMBDA > 0 and GATE_LAMBDA < 1:
            # 残差门控：R_proj = σ(λ)·TopK(R_raw) + (1-σ(λ))·TopK(RowNorm(A_dp) @ RowNorm(A_pd))
            R_residual = (row_norm(A_dp) @ row_norm(A_pd)).tocsr()
            
            # 应用门控权重
            R_proj_gated = (R_proj_raw * GATE_LAMBDA + R_residual * (1 - GATE_LAMBDA)).tocsr()
            print(f"[PROJ-GATE] 残差门控: λ={GATE_LAMBDA}, R_raw nnz={R_proj_raw.nnz}, R_residual nnz={R_residual.nnz}")
            R_proj_raw = R_proj_gated
        
        # === 使用LGCN对蛋白质投影进行降噪处理 ===
        # 注意：这里传入增强后的 R_obs，而不是原始的 train_csr
        R_proj = denoise_protein_projection_with_lgcn(R_proj_raw, self.num_drugs, self.num_diseases, R_obs)
        
        # === LGCN++优化：一致性门（后验过滤） ===
        CONSIST_GATE = int(os.environ.get("CONSIST_GATE", "0"))
        if CONSIST_GATE == 1:
            CONSIST_THR = float(os.environ.get("CONSIST_THR", "0.6"))
            CONSIST_NEI = int(os.environ.get("CONSIST_NEI", "10"))
            CONSIST_METHOD = os.environ.get("CONSIST_METHOD", "cross_threshold")
            
            print(f"[PROJ-CONSIST] 应用一致性门: THR={CONSIST_THR}, NEI={CONSIST_NEI}")
            
            # 这里需要相似图信息，暂时跳过具体实现
            # 在实际使用中，需要传入drug_sim和disease_sim
            print(f"[PROJ-CONSIST] 一致性门已启用，但需要相似图信息")
        
        # 严格屏蔽验证集和测试集正关联，防止投影边进入训练图或伪标签监督。
        held_out = (self.val_csr.astype(bool) + self.test_csr.astype(bool)).astype(bool)
        held_rows, held_cols = held_out.nonzero()
        projected_before_mask = R_proj.nnz
        if len(held_rows):
            R_proj = R_proj.tolil(copy=True)
            R_proj[held_rows, held_cols] = 0.0
            R_proj = R_proj.tocsr()
            R_proj.eliminate_zeros()
        print(
            f'[PROJ-LEAKAGE-GUARD] removed={projected_before_mask - R_proj.nnz}, '
            f'remaining={R_proj.nnz}'
        )

        # === 应用投影权重并合并 ===
        if self.proj_weight != 1.0:
            R_proj = R_proj.multiply(self.proj_weight)
        R_msg = (R_obs + R_proj).tocsr()
        print(f"[PROJ] R_obs nnz={R_obs.nnz}, R_msg nnz={R_msg.nnz}, added={R_msg.nnz - R_obs.nnz}")
        
        # （可选）调试：统计投影命中 test 的覆盖率
        if int(os.environ.get("PROJ_DEBUG", "0")) == 1:
            test_pairs = set((d, di) for d, dis in self.test_dict.items() for di in dis)
            proj_pairs = set(zip(*R_proj.nonzero()))
            hit = len(test_pairs & proj_pairs)
            cov = hit / (len(test_pairs) + 1e-8)
            print(f"[PROJ][DEBUG] coverage_on_test: hit={hit}, total_test={len(test_pairs)}, cov={cov:.4f}")

        # 6) 拼成 (D+Di)×(D+Di) 的对称方阵邻接（保留权重）
        coo = R_msg.tocoo()
        rows = np.concatenate([coo.row, coo.col + D])
        cols = np.concatenate([coo.col + D, coo.row])
        data = np.concatenate([coo.data, coo.data]).astype(np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(D + Di, D + Di)).tocsr()
        # === 保存二跳投影掩码，供 SSL_DEBUG 统计之用（D×Di 布尔CSR） ===
        try:
            R_obs_bool = (R_obs > 0).astype(np.int8)
            R_msg_bool = (R_msg > 0).astype(np.int8).tolil()
            ru, ci = R_obs_bool.nonzero()
            R_msg_bool[ru, ci] = 0  # 把观测边从“消息边集合”里剔掉，留下纯投影边
            self.R_proj_mask = R_msg_bool.tocsr().astype(bool)
        except Exception as e:
            print(f"[WARN] fail to build R_proj_mask: {e}")
            self.R_proj_mask = None

        # 保存仅由训练阶段信息产生、且已排除观测与留出正例的伪正样本候选。
        self.pseudo_pos_pairs = np.zeros((0, 2), dtype=np.int64)
        self.pseudo_pos_confidence = np.zeros(0, dtype=np.float32)
        if 'R_proj' in locals() and R_proj.nnz:
            candidates = R_proj.tocsr(copy=True)
            observed_rows, observed_cols = self.train_csr.nonzero()
            candidates = candidates.tolil()
            candidates[observed_rows, observed_cols] = 0.0
            candidates = candidates.tocsr()
            candidates.eliminate_zeros()
            candidate_coo = candidates.tocoo()
            self.pseudo_pos_pairs = np.stack(
                [candidate_coo.row, candidate_coo.col], axis=1
            ).astype(np.int64)
            confidence = np.asarray(candidate_coo.data, dtype=np.float32)
            confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
            confidence = np.maximum(confidence, 0.0)
            maximum = float(confidence.max()) if len(confidence) else 0.0
            if maximum > 0:
                confidence = confidence / maximum
            self.pseudo_pos_confidence = confidence.astype(np.float32)
        print(
            f'[PSEUDO] leakage-safe candidates={len(self.pseudo_pos_pairs)}, '
            f'fold={self.fold}'
        )

        return adj




    
    def create_adj_mat(self, is_subgraph=False, aug_type='ed', rng=None):
        """
        生成对比学习用的子图（分层抽样：观测 vs 投影）。
        环境变量：
          SSL_USE_MSG=1/0  : 是否用增强图采样（默认 1）
          SSL_DEG_SAFE=1/0 : 丢弃后零度兜底（默认 1）
          SSL_P_OBS=float  : 子图中观测边目标占比（默认 0.6）
          SSL_DEBUG=1      : 打印子图里投影边占比（需 self.R_proj_mask）
        说明：
          - 若 self.R_proj_mask 存在，则将增强图 D×Di 中的边分为 "观测" 与 "投影" 两层分别采样；
          - 若不存在，则退化为从 R_src 全集随机采样。
        """
        import numpy as np
        import scipy.sparse as sp
    
        if rng is None:
            rng = np.random.default_rng(self.seed)
        D, Di = self.num_drugs, self.num_diseases
        n_nodes = D + Di
    
        # ---------- 采样源：增强后的 D×Di（观测 ∪ 蛋白二跳）或原始 train_np ----------
        use_msg = bool(int(os.environ.get("SSL_USE_MSG", "1")))
        if use_msg:
            # D×Di
            R_src = self.adj_train_msg[:D, D:D+Di].tocsr()
            drugs_all, diseases_all = R_src.nonzero()
        else:
            drugs_all, diseases_all = self.train_np[:, 0], self.train_np[:, 1]
            R_src = sp.csr_matrix(
                (np.ones_like(drugs_all, np.float32), (drugs_all, diseases_all)),
                shape=(D, Di)
            )
    
        # ---------- 不增强，直接返回全集 ----------
        if not (is_subgraph and self.ssl_ratio > 0):
            data = np.ones_like(drugs_all, dtype=np.float32)
            tmp_adj = sp.csr_matrix((data, (drugs_all, diseases_all + D)), shape=(n_nodes, n_nodes))
            # 对称归一化
            A = tmp_adj + tmp_adj.T
            rowsum = np.asarray(A.sum(1)).ravel().astype(np.float32)
            rowsum[rowsum == 0] = 1.0
            D = sp.diags(rowsum ** -0.5)
            return D @ A @ D
    
        # ================== 做增强 ==================
        if aug_type == 'nd':
            # ---------- 节点丢弃（行/列屏蔽） ----------
            drop_d = rng.choice(D, size=int(D * self.ssl_ratio), replace=False) if D > 0 else np.array([], dtype=np.int64)
            drop_di = rng.choice(Di, size=int(Di * self.ssl_ratio), replace=False) if Di > 0 else np.array([], dtype=np.int64)
            keep_d = np.ones(D, dtype=bool); keep_d[drop_d] = False
            keep_di = np.ones(Di, dtype=bool); keep_di[drop_di] = False
    
            R_keep = R_src[keep_d, :][:, keep_di].tocsr()
            ru, ci = R_keep.nonzero()
            map_d = np.flatnonzero(keep_d)[ru]
            map_di = np.flatnonzero(keep_di)[ci]
            data = np.ones_like(map_d, dtype=np.float32)
            rows = np.concatenate([map_d, map_di + D])
            cols = np.concatenate([map_di + D, map_d])
            dat  = np.concatenate([data, data])
            tmp_adj = sp.coo_matrix((dat, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    
        else:
            # ---------- 边丢弃：分层（观测 vs 投影）抽样 + 多重兜底 ----------
            # 全集大小与目标保留
            m_total = len(drugs_all)
            keep_m  = max(1, int(round(m_total * (1.0 - self.ssl_ratio))))
            keep_m  = min(keep_m, m_total)  # 总不能超过全集
    
            # 分层候选
            if hasattr(self, "R_proj_mask") and (self.R_proj_mask is not None) and self.R_proj_mask.shape == (D, Di):
                # 观测掩码 = 全集 - 投影
                R_obs_mask = R_src.copy().tocsr()
                pm = self.R_proj_mask.tocoo()
                if pm.nnz > 0:
                    R_obs_mask[pm.row, pm.col] = 0.0
                    R_obs_mask.eliminate_zeros()
                ou, oi = R_obs_mask.nonzero()        # 观测候选
                pu, pi = self.R_proj_mask.nonzero()  # 投影候选
            else:
                # 无法区分时，全部当作“观测”
                ou, oi = R_src.nonzero()
                pu, pi = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
            m_obs, m_prj = len(ou), len(pu)
            p_obs = float(os.environ.get("SSL_P_OBS", "0.6"))
            p_obs = min(max(p_obs, 0.0), 1.0)
            # 原来：keep_m = int(round(m_total * (1.0 - self.ssl_ratio)))
            keep_m  = int(round(m_total * (1.0 - self.ssl_ratio)))
            # 关键：根据观测候选的天花板，收缩 keep_m，尽量达到 p_obs
            if p_obs > 0:
                keep_m = min(keep_m, int(m_obs / p_obs))   # 确保观测能占到 p_obs
            keep_m = max(1, keep_m)                        # 最少保 1

            # 目标配额
            k_obs_des = int(round(keep_m * p_obs))
            k_prj_des = keep_m - k_obs_des
    
            # 各自截断，并允许互借
            k_obs = min(k_obs_des, m_obs)
            k_prj = min(k_prj_des, m_prj)
            resid = keep_m - (k_obs + k_prj)
            if resid > 0 and m_obs - k_obs > 0:
                take = min(resid, m_obs - k_obs); k_obs += take; resid -= take
            if resid > 0 and m_prj - k_prj > 0:
                take = min(resid, m_prj - k_prj); k_prj += take; resid -= take
            # 如果仍有 resid>0，后面会用全集补齐
    
            # 安全不放回采样（不足时自动缩小到候选数）
            def safe_pick(n, k):
                if n <= 0 or k <= 0:
                    return np.empty(0, dtype=np.int64)
                k = min(int(k), int(n))
                return rng.choice(n, size=k, replace=False).astype(np.int64)
    
            idx_o = safe_pick(m_obs, k_obs)
            idx_p = safe_pick(m_prj, k_prj)
    
            su, si = (ou[idx_o], oi[idx_o]) if idx_o.size else (np.empty(0, np.int64), np.empty(0, np.int64))
            tu, ti = (pu[idx_p], pi[idx_p]) if idx_p.size else (np.empty(0, np.int64), np.empty(0, np.int64))
    
            drug_np = np.concatenate([su, tu])
            disease_np = np.concatenate([si, ti])
    
            # 仍不足则从全集“放回采样”补齐并去重
            short = keep_m - drug_np.size
            if short > 0 and m_total > 0:
                add_idx = rng.choice(m_total, size=short * 2, replace=True)  # 多取点备去重
                cand_d  = np.asarray(drugs_all)[add_idx]
                cand_di = np.asarray(diseases_all)[add_idx]
                chosen = set(zip(drug_np.tolist(), disease_np.tolist()))
                extra_d, extra_di = [], []
                for dd, ii in zip(cand_d, cand_di):
                    if (dd, ii) not in chosen:
                        chosen.add((dd, ii))
                        extra_d.append(dd); extra_di.append(ii)
                        if len(extra_d) >= short:
                            break
                if extra_d:
                    drug_np = np.concatenate([drug_np, np.asarray(extra_d, np.int64)])
                    disease_np = np.concatenate([disease_np, np.asarray(extra_di, np.int64)])
    
            # 零度兜底（可选）
            if int(os.environ.get("SSL_DEG_SAFE", "1")):
                R_keep = sp.csr_matrix(
                    (np.ones_like(drug_np, np.float32), (drug_np, disease_np)),
                    shape=(D, Di)
                ).tolil()
    
                # 用户侧
                deg_u = np.asarray(R_keep.sum(1)).ravel()
                for u in np.where(deg_u == 0)[0]:
                    nbrs = R_src.getrow(u).indices
                    if len(nbrs) > 0:
                        R_keep[u, rng.choice(nbrs)] = 1.0
    
                # 疾病侧
                deg_i = np.asarray(R_keep.sum(0)).ravel()
                for i in np.where(deg_i == 0)[0]:
                    nbrs = R_src.getcol(i).indices
                    if len(nbrs) > 0:
                        R_keep[rng.choice(nbrs), i] = 1.0
    
                R_keep = R_keep.tocsr()
                drug_np, disease_np = R_keep.nonzero()
    
            data = np.ones_like(drug_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((data, (drug_np, disease_np + D)), shape=(n_nodes, n_nodes))
    
        # ---------- DEBUG：子图里投影边占比 ----------
        if int(os.environ.get("SSL_DEBUG", "0")) and use_msg and hasattr(self, "R_proj_mask") and \
           (self.R_proj_mask is not None) and self.R_proj_mask.shape == (D, Di):
            R_view = tmp_adj[:D, D:D+Di].tocsr()
            hits = int((R_view.multiply(self.R_proj_mask)).sum())
            tot  = int(R_view.count_nonzero())
            rate = (hits / tot) if tot > 0 else 0.0
            print(f"[SSL-DEBUG] subgraph edges={tot}, from projection={hits} ({rate:.2%})")
    
        # ---------- 对称化 + 对称归一化 ----------
        A = tmp_adj + tmp_adj.T
        rowsum = np.asarray(A.sum(1)).ravel().astype(np.float32)
        rowsum[rowsum == 0] = 1.0
        D = sp.diags(rowsum ** -0.5)
        return D @ A @ D


    def load_pickle(self, name):
        with open(name, 'rb') as f:
            return pkl.load(f, encoding='latin1')

    def convert_to_inner_index(self, drug_records, drug_mapping, disease_mapping):
        inner_drug_records = []
        drug_inverse_mapping = self.generate_inverse_mapping(drug_mapping)
        disease_inverse_mapping = self.generate_inverse_mapping(disease_mapping)

        for drug_id in range(len(drug_mapping)):
            real_drug_id = drug_mapping[drug_id]
            disease_list = list(drug_records[real_drug_id])
            for index, real_disease_id in enumerate(disease_list):
                disease_list[index] = disease_inverse_mapping[real_disease_id]
            inner_drug_records.append(disease_list)

        return inner_drug_records, drug_inverse_mapping, disease_inverse_mapping

    def generate_inverse_mapping(self, mapping):
        inverse_mapping = dict()
        for inner_id, true_id in enumerate(mapping):
            inverse_mapping[true_id] = inner_id
        return inverse_mapping

    def generate_rating_matrix(self, train_set, num_drugs, num_diseases): 
        row = []
        col = []
        data = []
        for drug_id, disease_list in enumerate(train_set):
            for disease in disease_list:
                row.append(drug_id)
                col.append(disease)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_drugs, num_diseases))
        return rating_matrix
        
    def _build_ban_and_neg_pool(self, pool_topk: int = 50):
        """
        为每个 drug(d) 构建分层Ban集合：
          - ban_observed[d]: 训练集已观测的真阳性疾病集合（永不缩减）
          - ban_soft[d]: 投影/去噪/消息传递的软边疾病集合（允许缩减）
          - ban_diseases[d]: 两者的并集（兼容旧代码）
          - hard_neg_pool[d]: 候选"困难负样本池"（来自与正例相似的疾病）
        """
        D, Di = self.num_drugs, self.num_diseases
    
        # === 分层 Ban 集合 ===
        ban_observed = [set() for _ in range(D)]
        ban_soft = [set() for _ in range(D)]
        
        # 1) 观测正例 -> ban_observed（永不缩减）
        for split_dict in (self.train_dict, self.val_dict, self.test_dict):
            for d, diseases in split_dict.items():
                for di in diseases:
                    ban_observed[d].add(int(di))
        
        # 2) 消息传递/投影/去噪得到的软边 -> ban_soft（可缩减）
        R_all = self.adj_train_msg[:D, D:D+Di].astype(bool).tocsr()
        for d in range(D):
            cols = R_all.getrow(d).indices
            for di in cols:
                if int(di) not in ban_observed[d]:  # 避免重复
                    ban_soft[d].add(int(di))
        
        # 3) 方便兼容旧代码：并集仍然提供
        ban_diseases = [ban_observed[d] | ban_soft[d] for d in range(D)]
    
        # === 困难池构建 ===
        # 加载疾病相似性矩阵
        disease_sim_path = os.path.join(self.pkl_path, 'DiseasePS.csv')  # 可换为 DiseaseGIP.csv
        disease_sim = pd.read_csv(disease_sim_path, index_col=0).values.astype(np.float32)  # Di×Di
        np.fill_diagonal(disease_sim, 0.0)
        
        # 加载药物相似性矩阵
        drug_fp_path = os.path.join(self.pkl_path, 'DrugFingerprint.csv')  # 药物指纹相似性
        drug_gip_path = self._gip_path('drug')  # 当前折训练关联计算的药物 GIP
        
        if os.path.exists(drug_fp_path):
            drug_fp_sim = pd.read_csv(drug_fp_path, index_col=0).values.astype(np.float32)  # D×D
            np.fill_diagonal(drug_fp_sim, 0.0)
            self.S_drug_fp = drug_fp_sim
            print(f"[SIM] 药物指纹相似性矩阵: {drug_fp_sim.shape}")
        else:
            self.S_drug_fp = None
            
        if os.path.exists(drug_gip_path):
            drug_gip_sim = self._read_similarity_matrix(drug_gip_path, D)  # D×D
            np.fill_diagonal(drug_gip_sim, 0.0)
            self.S_drug_gip = drug_gip_sim
            print(f"[SIM] 药物高斯相似性矩阵: {drug_gip_sim.shape}")
        else:
            self.S_drug_gip = None
        
        # 保存疾病相似性矩阵供后续使用
        self.S_disease = disease_sim
        print(f"[SIM] 疾病相似性矩阵: {disease_sim.shape}")
    
        hard_pool = []
        for d in range(D):
            seeds = list(self.train_dict.get(d, []))
            if not seeds:
                deg = np.asarray(self.train_csr.sum(0)).ravel()
                order = deg.argsort()[::-1]
                cand = [int(di) for di in order if int(di) not in ban_diseases[d]]
                hard_pool.append(np.array(cand[:pool_topk], dtype=np.int32))
                continue
    
            score = np.zeros(Di, dtype=np.float32)
            for i in seeds:
                score += disease_sim[i]
            # 去掉 ban（使用分层Ban）
            for di in ban_diseases[d]:
                score[di] = -np.inf
            order = np.argsort(-score)
            top = [int(i) for i in order[:pool_topk] if np.isfinite(score[i])]
            hard_pool.append(np.array(top, dtype=np.int32))
    
        # 保存分层Ban集合
        self.ban_observed = ban_observed
        self.ban_soft = ban_soft
        self.ban_diseases = ban_diseases
        self.hard_neg_pool = hard_pool
        print(f"[HARDNEG] built: pool_topk={pool_topk}")
        print(f"[BAN] ban_observed: {sum(len(s) for s in ban_observed)} total")
        print(f"[BAN] ban_soft: {sum(len(s) for s in ban_soft)} total")

    def reduce_ban_by_similarity(self, ban_soft_set, seeds, factor: float):
        """
        按相似性缩减：保留 factor 比例【相似度更高】的软边在 Ban（不释放），释放低相似的。
        seeds: 当前药物的正例疾病列表
        factor: 保留比例 (0.0-1.0)
        """
        if not ban_soft_set or not seeds:
            return ban_soft_set
        
        ban_list = list(ban_soft_set)  # set 不能切片，必须转 list
        
        # importance = max_{i in seeds} S_disease[i, di]
        imp = []
        for di in ban_list:
            imp.append(float(self.S_disease[seeds, di].max()))
        
        imp = np.asarray(imp)
        order = np.argsort(-imp)  # 高相似优先保留
        keep = max(1, int(len(ban_list) * factor))
        kept = {ban_list[i] for i in order[:keep]}
        return kept

    def reduce_ban_random(self, ban_soft_set, factor: float, rng):
        """随机缩减：保留 factor 比例的元素在 Ban 中"""
        if not ban_soft_set:
            return ban_soft_set
        
        ban_list = list(ban_soft_set)
        rng.shuffle(ban_list)
        keep = max(1, int(len(ban_list) * factor))
        return set(ban_list[:keep])

    def reduce_ban_by_importance(self, ban_soft_set, importance_vec, factor: float):
        """
        通用重要性缩减：importance_vec[di] 越大越该保留在 Ban。
        你可以把 '药物相似投票'、'结构接近度' 融进去当 importance。
        """
        if not ban_soft_set:
            return ban_soft_set
        
        ban_list = list(ban_soft_set)
        imp = np.asarray([importance_vec[di] for di in ban_list], dtype=np.float32)
        order = np.argsort(-imp)
        keep = max(1, int(len(ban_list) * factor))
        keep_idx = order[:keep]
        kept = {ban_list[i] for i in keep_idx}
        return kept

    def adaptive_negative_sampling(self, drugs, rng, hard_pool, 
                                   use_similarity_shrink=True,
                                   shrink_factor=0.7, 
                                   max_attempts=10):
        """
        优化版自适应负采样：添加缓存和简化逻辑
        """
        # 使用缓存避免重复计算
        if not hasattr(self, '_neg_sampling_cache'):
            self._neg_sampling_cache = {}
        
        all_diseases = np.arange(self.num_diseases, dtype=np.int32)
        deg = np.asarray(self.train_csr.sum(0)).ravel()
        
        # 长尾校正参数
        ETA = getattr(self, "LONGTAIL_ETA", 0.5)
        
        chosen_list = []
        
        for d in drugs:
            chosen = None
            
            # 检查缓存
            cache_key = f"{d}_{len(self.ban_soft[d])}_{shrink_factor}"
            if cache_key in self._neg_sampling_cache:
                chosen = self._neg_sampling_cache[cache_key]
                if chosen is not None and chosen not in self.ban_observed[d]:
                    chosen_list.append(chosen)
                    continue
            
            # --------- (1) 困难池优先（简化版） ----------
            pool = hard_pool[d]
            if pool is not None and len(pool) > 0:
                ban_obs = self.ban_observed[d]
                # 只尝试5次，不是10次
                for _ in range(5):
                    di = int(rng.choice(pool))
                    if di not in ban_obs:
                        chosen = di
                        break
            
            # --------- (2) 全集 \ Ban（简化版） ----------
            if chosen is None:
                ban_all = self.ban_observed[d] | self.ban_soft[d]
                if len(ban_all) < self.num_diseases * 0.8:  # 只有Ban集合不太大时才用这个方法
                    ban_mask = np.zeros(self.num_diseases, dtype=bool)
                    ban_idx = np.fromiter(ban_all, dtype=np.int32)
                    ban_mask[ban_idx] = True
                    candidate = all_diseases[~ban_mask]
                    if candidate.size > 0:
                        # 简化权重计算
                        w = (deg[candidate] + 1e-3) ** (-ETA)
                        w = w / w.sum()
                        chosen = int(rng.choice(candidate, p=w))
            
            # --------- (3) 简化Ban缩减 ----------
            if chosen is None and len(self.ban_soft[d]) > 0:
                # 只缩减一次，不重复计算相似度
                if use_similarity_shrink and hasattr(self, "S_disease") and len(self.train_dict.get(int(d), [])) > 0:
                    # 使用预计算的相似度
                    seeds = list(self.train_dict.get(int(d), []))
                    if len(seeds) > 0:
                        new_soft = self.reduce_ban_by_similarity(
                            self.ban_soft[d], seeds=seeds, factor=shrink_factor
                        )
                    else:
                        new_soft = self.reduce_ban_random(self.ban_soft[d], factor=shrink_factor, rng=rng)
                else:
                    new_soft = self.reduce_ban_random(self.ban_soft[d], factor=shrink_factor, rng=rng)
                
                # 更新Ban集合（只更新一次）
                self.ban_soft[d] = new_soft
                ban_all = self.ban_observed[d] | new_soft
                
                if len(ban_all) < self.num_diseases * 0.9:
                    ban_mask = np.zeros(self.num_diseases, dtype=bool)
                    ban_idx = np.fromiter(ban_all, dtype=np.int32) if len(ban_all) else np.array([], dtype=np.int32)
                    if ban_idx.size > 0:
                        ban_mask[ban_idx] = True
                    candidate = all_diseases[~ban_mask]
                    if candidate.size > 0:
                        w = (deg[candidate] + 1e-3) ** (-ETA)
                        w = w / w.sum()
                        chosen = int(rng.choice(candidate, p=w))
            
            # --------- (4) 兜底随机（简化版） ----------
            if chosen is None:
                ban_obs = self.ban_observed[d]
                # 只尝试50次，不是200次
                for _ in range(50):
                    di = int(rng.integers(0, self.num_diseases))
                    if di not in ban_obs:
                        chosen = di
                        break
                
                if chosen is None:
                    # 最后兜底
                    chosen = int(rng.integers(0, self.num_diseases))
            
            # 缓存结果
            self._neg_sampling_cache[cache_key] = chosen
            chosen_list.append(chosen)
        
        return np.asarray(chosen_list, dtype=np.int32)

    def fast_negative_sampling(self, drugs, rng, hard_pool):
        """
        快速负采样：简化版本，优先速度
        """
        all_diseases = np.arange(self.num_diseases, dtype=np.int32)
        chosen_list = []
        
        for d in drugs:
            chosen = None
            
            # 1) 困难池优先（只尝试3次）
            pool = hard_pool[d]
            if pool is not None and len(pool) > 0:
                ban_obs = self.ban_observed[d]
                for _ in range(3):
                    di = int(rng.choice(pool))
                    if di not in ban_obs:
                        chosen = di
                        break
            
            # 2) 随机采样（避开观测正例）
            if chosen is None:
                ban_obs = self.ban_observed[d]
                for _ in range(20):
                    di = int(rng.integers(0, self.num_diseases))
                    if di not in ban_obs:
                        chosen = di
                        break
                
                # 最后兜底
                if chosen is None:
                    chosen = int(rng.integers(0, self.num_diseases))
            
            chosen_list.append(chosen)
        
        return np.asarray(chosen_list, dtype=np.int32)

    def load_similarity_graphs(self):
        SIM_TOPK = int(os.environ.get("SIM_TOPK", "40"))   # 每行保留的相似边数；0 表示不裁剪
        DROP_SELF = int(os.environ.get("SIM_DROP_SELF", "1"))  # 是否去掉自环（默认去）
    
        def load_and_normalize(path, expected_dim):
            import os
            mat = self._read_similarity_matrix(path, expected_dim)
            sp_mat = sp.csr_matrix(mat)
    
            # 1) 对称化（有些相似度矩阵可能非完全对称）
            sp_mat = sp_mat.maximum(sp_mat.T)
    
            # 2) 去掉对角（自相似），避免一行只剩自环顶掉 TopK
            if DROP_SELF:
                sp_mat.setdiag(0.0)
                sp_mat.eliminate_zeros()
    
            # 3) 行 Top-K（保留每行最强的 K 条相似边）
            if SIM_TOPK > 0:
                sp_mat = self._row_topk(sp_mat, SIM_TOPK)
    
            # 4) 对称归一化  D^{-1/2} A D^{-1/2}
            rowsum = np.array(sp_mat.sum(1)).flatten()
            rowsum[rowsum == 0] = 1.0
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_mat = d_mat_inv_sqrt @ sp_mat @ d_mat_inv_sqrt
    
            # 可选：打印一下压缩率
            print(f"[SIM] {os.path.basename(path)} nnz={sp_mat.nnz} (topk={SIM_TOPK}, drop_self={DROP_SELF})")
            return sparse_mx_to_torch_sparse_tensor(norm_mat)
    
        self.drug_gip_sim    = load_and_normalize(self._gip_path('drug'), self.num_drugs)
        self.disease_gip_sim = load_and_normalize(self._gip_path('disease'), self.num_diseases)
        self.drug_fp_sim     = load_and_normalize(os.path.join(self.pkl_path, 'DrugFingerprint.csv'), self.num_drugs)
        self.disease_ps_sim  = load_and_normalize(os.path.join(self.pkl_path, 'DiseasePS.csv'),       self.num_diseases)
    
        print("✔ Loaded and normalized auxiliary similarity graphs:")
        print(f"  drug_gip_sim: {self.drug_gip_sim.shape}")
        print(f"  disease_gip_sim: {self.disease_gip_sim.shape}")
        print(f"  drug_fp_sim: {self.drug_fp_sim.shape}")
        print(f"  disease_ps_sim: {self.disease_ps_sim.shape}")

    
