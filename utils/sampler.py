from scipy.sparse import lil_matrix, csr_matrix
from multiprocessing import Process, Queue

import numpy as np
import itertools


# ========== 新增：多进程worker：使用候选池进行负采样 ==========
def sample_function(drug_disease_matrix,
                    num_nodes,
                    batch_size,
                    n_negative,
                    result_queue,
                    candidate_diseases,
                    seed):
    """
    子进程采样：
      - 正样本从观测图 (D×Di) 抽取；
      - 负样本仅从每个药物的允许候选池里采样（已避开 观测∪投影TopK ）。
    """
    num_drugs, num_diseases = num_nodes
    adj_train = lil_matrix(drug_disease_matrix)

    # 仅保留 D->Di 的正边（行是 drug，列是 disease 节点= offset + disease_id）
    rows, cols = adj_train.nonzero()
    mask = (rows < num_drugs) & (cols >= num_drugs)
    drug_disease_pairs = np.vstack((rows[mask], cols[mask])).T

    if len(drug_disease_pairs) == 0:
        raise RuntimeError("No drug-disease positive pairs found in training adjacency.")

    rng = np.random.default_rng(seed)

    while True:
        # 随机抽 batch_size 个正样本
        idx = rng.integers(0, len(drug_disease_pairs), size=batch_size)
        drug_positive_diseases_pairs = drug_disease_pairs[idx]

        # 负样本：对每个 d，从候选池采 n_negative 个（可重复）
        neg = np.empty((batch_size, n_negative), dtype=np.int64)
        for j in range(batch_size):
            d = int(drug_positive_diseases_pairs[j, 0])
            cand = candidate_diseases[d]
            if cand.size == 0:
                # 兜底：全体 disease 节点（仍是 offset 过的列索引）
                cand = np.arange(num_drugs, num_drugs + num_diseases, dtype=np.int64)
            neg[j] = rng.choice(cand, size=n_negative, replace=True)

        triples = np.hstack((drug_positive_diseases_pairs.astype(np.int64), neg))
        # 打乱顺序（可选）
        rng.shuffle(triples)
        result_queue.put(triples)


class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: drug-positive-disease pairs, negative-diseases
    """

    # ========== 改动1：新增 ban_csr 参数 ==========
    def __init__(self,
                 num_nodes,
                 drug_disease_matrix,
                 batch_size=10000,
                 n_negative=10,
                 n_workers=5,
                 ban_csr=None,
                 seed=0):
        """
        num_nodes: (num_drugs, num_diseases)
        drug_disease_matrix: 方阵 (D+Di)×(D+Di) 的观测邻接（对称）
        ban_csr: 可选，D×Di 的 CSR，表示“禁止作为负样本”的位置（观测 ∪ 增强TopK）
        """
        self.seed = int(seed)
        self.num_nodes = num_nodes
        self.drug_disease_matrix = drug_disease_matrix
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.n_workers = n_workers

        self.result_queue = Queue(maxsize=n_workers * 2) if n_workers > 0 else None
        self.processors = []

        # ======= 预处理：构建观测正边 + 禁止集合 + 候选池 =======
        self._prepare_candidates(ban_csr)

        if n_workers > 0:
            for i in range(n_workers):
                self.processors.append(
                    Process(
                        target=sample_function,
                        args=(
                            drug_disease_matrix,
                            self.seed + i,
                            num_nodes,
                            batch_size,
                            n_negative,
                            self.result_queue,
                            self._candidate_diseases,   # 传入候选池
                        )
                    )
                )
                self.processors[-1].start()
        else:
            # 单进程 fallback
            self._single_iter = self._single_process_generator()

    def next_batch(self):
        if self.n_workers > 0:
            return self.result_queue.get()
        else:
            return next(self._single_iter)

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

    # ========== 改动2：集中准备禁止集合 & 候选池 ==========
    def _prepare_candidates(self, ban_csr):
        num_drugs, num_diseases = self.num_nodes
        adj_train = lil_matrix(self.drug_disease_matrix)

        # 观测正边：药物行的邻接集合（列索引是 disease 节点的“节点下标”，已offset）
        # adj_train.rows[d] 是 lil 的行邻居列表
        obs_sets = {d: set(adj_train.rows[d]) for d in range(num_drugs)}

        # ban: 观测 ∪ 投影TopK（若传入 ban_csr 为 D×Di，则要把列索引 + num_drugs 转成节点下标）
        if ban_csr is not None:
            ban_csr = ban_csr.tocsr()
            self.ban_sets = {}
            for d in range(num_drugs):
                banned_diseases = ban_csr.getrow(d).indices  # 0..Di-1 的疾病列号
                # 转成节点下标（offset by num_drugs）
                banned_nodes = set((banned_diseases + num_drugs).tolist())
                self.ban_sets[d] = obs_sets[d] | banned_nodes
        else:
            # 至少禁止观测正边
            self.ban_sets = obs_sets

        # 为每个药物构建允许候选池（disease 节点的全体：num_drugs .. num_drugs+num_diseases-1）
        all_disease_nodes = np.arange(num_drugs, num_drugs + num_diseases, dtype=np.int64)
        self._candidate_diseases = []
        for d in range(num_drugs):
            forbid = self.ban_sets[d]
            if len(forbid) >= len(all_disease_nodes):
                cand = all_disease_nodes
            else:
                # 用布尔掩码更快
                mask = np.ones(all_disease_nodes.shape[0], dtype=bool)
                # 把 forbid 中位于 disease 区间的索引映射到 [0..num_diseases-1] 的位置再置 False
                for node_idx in forbid:
                    if num_drugs <= node_idx < num_drugs + num_diseases:
                        mask[node_idx - num_drugs] = False
                cand = all_disease_nodes[mask]
            self._candidate_diseases.append(cand)

    # ========== 改动3：单进程生成器使用候选池 ==========
    def _single_process_generator(self):
        num_drugs, num_diseases = self.num_nodes
        adj_train = lil_matrix(self.drug_disease_matrix)

        # 仅保留 D->Di 的正边
        rows, cols = adj_train.nonzero()
        mask = (rows < num_drugs) & (cols >= num_drugs)
        drug_disease_pairs = np.vstack((rows[mask], cols[mask])).T

        if len(drug_disease_pairs) == 0:
            raise RuntimeError("No drug-disease positive pairs found in training adjacency.")

        rng = np.random.default_rng(self.seed)

        while True:
            # 抽一个 batch 的正样本
            total = len(drug_disease_pairs)
            # 保证能整 batch 抽样
            n_batches = max(1, total // self.batch_size)
            for i in range(n_batches):
                idx = rng.integers(0, total, size=self.batch_size)
                drug_positive_diseases_pairs = drug_disease_pairs[idx]

                # 负样本：候选池采样（可重复）
                drug_negative_samples = np.empty((self.batch_size, self.n_negative), dtype=np.int64)
                for j, (d, _) in enumerate(drug_positive_diseases_pairs):
                    d = int(d)
                    cand = self._candidate_diseases[d]
                    if cand.size == 0:
                        cand = np.arange(num_drugs, num_drugs + num_diseases, dtype=np.int64)
                    drug_negative_samples[j] = rng.choice(cand, size=self.n_negative, replace=True)

                drug_triples = np.hstack((drug_positive_diseases_pairs.astype(np.int64),
                                          drug_negative_samples))
                rng.shuffle(drug_triples)
                yield drug_triples
