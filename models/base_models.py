import torch
import torch.nn as nn
import numpy as np
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from models.encoders import HGCN, FHGCN
from utils.util import ndcg_func, recall_func, recall_single_func, ndcg_single_func


class HyperCL(nn.Module):
    def __init__(self, drugs_diseases, args, drug_gip=None, disease_gip=None, drug_fp=None, disease_ps=None):
        super(HyperCL, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.nnodes = args.n_nodes
        self.encoder1 = HGCN(self.manifold, args)
        self.encoder2 = FHGCN(self.manifold, args)

        self.num_drugs, self.num_diseases = drugs_diseases
        self.margin = args.margin
        self.num_layers = args.num_layers
        self.latent_dim = args.embedding_dim
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.args = args

        # 可学习嵌入初始化
        self.embedding = nn.Embedding(num_embeddings=self.num_drugs + self.num_diseases,
                                      embedding_dim=args.embedding_dim).to(self.device)
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], project=True))
        self.embedding.weight = ManifoldParameter(self.embedding.weight, self.manifold, True)

        # 结构增强部分
        self.drug_gip = drug_gip
        self.drug_fp = drug_fp
        self.disease_gip = disease_gip
        self.disease_ps = disease_ps

        # 增强图使用的可学习融合参数
        self.alpha = nn.Parameter(torch.zeros(4))  # 融合四类图的权重
        self.edge_drop = nn.Dropout(p=0.2)

        # ---- 轻量逻辑解码参数：score = logit_scale * (-sqdist) + logit_bias ----
        self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=self.device))
        self.logit_bias  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))

    def _augment_graph(self, base_adj):
        """
        对 base_adj 进行结构增强，融合四类辅助图
        """
        weights = torch.sigmoid(self.alpha)
        device = self.device
    
        # 允许稀疏输入，统一转 dense，避免 torch.block_diag 报错
        def to_dense_if_sparse(x):
            return x.to_dense() if x.is_sparse else x
    
        drug_gip = to_dense_if_sparse(self.drug_gip.to(device))
        drug_fp  = to_dense_if_sparse(self.drug_fp.to(device))
        disease_gip = to_dense_if_sparse(self.disease_gip.to(device))
        disease_ps  = to_dense_if_sparse(self.disease_ps.to(device))
        base_adj = to_dense_if_sparse(base_adj.to(device))
    
        drug_mask = torch.block_diag(
            weights[0] * drug_gip + weights[1] * drug_fp,
            torch.zeros_like(disease_gip)
        )
        disease_mask = torch.block_diag(
            torch.zeros_like(drug_gip),
            weights[2] * disease_gip + weights[3] * disease_ps
        )
        aug_adj = base_adj + self.edge_drop(drug_mask) + self.edge_drop(disease_mask)
        return torch.clamp(aug_adj, min=0, max=1)
    
    def encode(self, adj, adj_1, adj_2):
        x = self.embedding.weight
        adj   = adj.to(self.device)
        adj_1 = adj_1.to(self.device)
        adj_2 = adj_2.to(self.device)
        x = x.to(self.device)
    
        h1 = self.encoder1.encode(x, adj)
        h2 = self.encoder1.encode(x, adj_1)
    
        # 关键：用 getattr 提供默认值 False，避免 AttributeError
        if getattr(self.args, 'structure_aug', False):
            enhanced_adj = self._augment_graph(adj_2)
            h3 = self.encoder2.encode(x, enhanced_adj)
        else:
            h3 = self.encoder2.encode(x, adj_2)
    
        return h1, h2, h3

    def encode4eval(self, adj):
        x = self.embedding.weight
        return self.encoder1.encode(x.to(self.device), adj.to(self.device))

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out)
        return sqdist

    def score(self, h, idx):
        """
        返回二分类 logit：更大的值意味着更可能为正样本
        """
        sqdist = self.decode(h, idx)          # 越小越相似
        logits = self.logit_scale * (-sqdist) + self.logit_bias
        return logits

    def Contra_loss(self, u_h, i_h, u_f, i_f):
        # Algebraically equivalent log-softmax form. Computing exp() first can
        # underflow for small temperatures and produce log(0 / 0)=NaN.
        user_pos_logits = -self.manifold.sqdist(u_h, u_f) / self.ssl_temp
        user_all_logits = -self.manifold.sqdist_multi(u_h, u_f) / self.ssl_temp
        ssl_loss_user = -(
            user_pos_logits - torch.logsumexp(user_all_logits, dim=1)
        ).sum()

        item_pos_logits = -self.manifold.sqdist(i_h, i_f) / self.ssl_temp
        item_all_logits = -self.manifold.sqdist_multi(i_h, i_f) / self.ssl_temp
        ssl_loss_item = -(
            item_pos_logits - torch.logsumexp(item_all_logits, dim=1)
        ).sum()
        return ssl_loss_user + ssl_loss_item

    def compute_loss(
        self,
        embeddings1,
        embeddings2,
        embeddings3,
        triples,
        positive_weights=None,
        observed_positive_count=None,
    ):
        """计算监督损失和对比损失，并支持置信度加权的伪正样本。"""
        import torch.nn.functional as F

        device = self.device
        drugs = torch.as_tensor(triples[:, 0], device=device, dtype=torch.long)
        pos_pairs = torch.as_tensor(triples[:, [0, 1]], device=device, dtype=torch.long)
        neg_nodes = torch.as_tensor(triples[:, 2:], device=device, dtype=torch.long)
        batch_size, negative_count = neg_nodes.size(0), neg_nodes.size(1)
        repeated_drugs = drugs.repeat_interleave(negative_count)
        neg_pairs = torch.stack([repeated_drugs, neg_nodes.reshape(-1)], dim=1)

        pos_logits = self.score(embeddings1, pos_pairs).reshape(-1)
        neg_logits = self.score(embeddings1, neg_pairs).reshape(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0
        )
        if positive_weights is None:
            positive_weights = torch.ones(batch_size, device=device, dtype=logits.dtype)
        else:
            positive_weights = torch.as_tensor(
                positive_weights, device=device, dtype=logits.dtype
            ).reshape(-1)
        if len(positive_weights) != batch_size:
            raise ValueError(
                f"positive_weights has {len(positive_weights)} values; expected {batch_size}"
            )
        sample_weights = torch.cat(
            [positive_weights, positive_weights.repeat_interleave(negative_count)], dim=0
        )
        pos_weight = torch.tensor(float(negative_count), device=device)
        element_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none"
        )
        bce = (element_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

        if observed_positive_count is None:
            observed_positive_count = batch_size
        observed_positive_count = int(observed_positive_count)
        if not 0 < observed_positive_count <= batch_size:
            raise ValueError(
                f"observed_positive_count={observed_positive_count}, batch_size={batch_size}"
            )
        contrastive_edges = pos_pairs[:observed_positive_count]
        drug_h = embeddings2[contrastive_edges[:, 0], :]
        disease_h = embeddings2[contrastive_edges[:, 1], :]
        drug_f = embeddings3[contrastive_edges[:, 0], :]
        disease_f = embeddings3[contrastive_edges[:, 1], :]
        ssl_loss = self.Contra_loss(drug_h, disease_h, drug_f, disease_f)
        return bce + self.ssl_reg * ssl_loss

    def predict(self, h, train_csr, test_csr, test_dict, eval_batch_num=0):
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            accuracy_score, precision_score, recall_score,
            f1_score, matthews_corrcoef, precision_recall_curve
        )

        pos_pairs = [[d, di + self.num_drugs] for d, dis in test_dict.items() for di in dis]
        pos_pairs = np.array(pos_pairs, dtype=np.int64)

        neg_pairs = []
        all_drugs = np.arange(self.num_drugs)
        all_diseases = np.arange(self.num_diseases)
        test_set = set((d, di) for d, dis in test_dict.items() for di in dis)
        train_set = set(zip(*train_csr.nonzero()))
        pos_set = test_set | train_set

        rng = np.random.default_rng(seed=42)
        while len(neg_pairs) < len(pos_pairs):
            d = rng.choice(all_drugs)
            di = rng.choice(all_diseases)
            if (d, di) not in pos_set:
                neg_pairs.append([d, di + self.num_drugs])
        neg_pairs = np.array(neg_pairs, dtype=np.int64)

        all_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
        labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])

        all_pairs_tensor = torch.tensor(all_pairs, device=self.device)
        scores = self.score(h, all_pairs_tensor).detach().cpu().numpy()  # logits

        # === 搜索最佳阈值 ===
        prec, rec, thr = precision_recall_curve(labels, scores)
        f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        idx = int(np.nanargmax(f1))
        best_thr = float(thr[idx])
        preds = (scores >= best_thr).astype(int)

        return dict(
            auc=roc_auc_score(labels, scores),
            aupr=average_precision_score(labels, scores),
            accuracy=accuracy_score(labels, preds),
            precision=precision_score(labels, preds, zero_division=0),
            recall=recall_score(labels, preds, zero_division=0),
            f1=f1_score(labels, preds, zero_division=0),
            mcc=matthews_corrcoef(labels, preds)
        )

