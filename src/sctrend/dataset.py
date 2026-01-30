import torch
import numpy as np

class ScDataSet(torch.utils.data.Dataset):
    def __init__(self, x, xnorm_mat, batch_onehot):
        self.x = x
        self.xnorm_mat = xnorm_mat
        self.batch_onehot = batch_onehot

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx_x = self.x[idx]
        idx_xnorm_mat = self.xnorm_mat[idx]
        idx_batch_onehot = self.batch_onehot[idx]
        return (idx_x, idx_xnorm_mat, idx_batch_onehot)

class ScDataManager():
    def __init__(self, x_count, batch_size, batch_onehot):
        validation_ratio = 0.1
        test_ratio = 0.05

        x_count = x_count.float()
        self.batch_onehot = batch_onehot
        xnorm_mat = torch.mean(x_count, dim=1).view(-1, 1)
        total_num = x_count.size()[0]
        self.x_count = x_count
        self.xnorm_mat = xnorm_mat

        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)

        np.random.seed(42)
        idx = np.random.permutation(np.arange(total_num))
        self.idx = torch.tensor(idx)

        validation_idx, test_idx, train_idx = (
            idx[:validation_num],
            idx[validation_num:(validation_num + test_num)],
            idx[(validation_num + test_num):],
        )
        self.validation_idx = torch.tensor(validation_idx)
        self.test_idx = torch.tensor(test_idx)
        self.train_idx = torch.tensor(train_idx)

        self.train_x = x_count[train_idx]
        self.train_xnorm_mat = xnorm_mat[train_idx]
        self.validation_x = x_count[validation_idx]
        self.validation_xnorm_mat = xnorm_mat[validation_idx]
        self.test_x = x_count[test_idx]
        self.test_xnorm_mat = xnorm_mat[test_idx]
        self.train_batch_onehot = batch_onehot[train_idx]
        self.validation_batch_onehot = batch_onehot[validation_idx]
        self.test_batch_onehot = batch_onehot[test_idx]
        self.train_eds = ScDataSet(x_count[train_idx], xnorm_mat[train_idx], batch_onehot[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
        )

    def initialize_loader(self, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
        )

class BulkDataManager():
    def __init__(self, bulk_count, survival_time, cutting_off_0_1):
        bulk_count = bulk_count.float()
        survival_time = survival_time.float()
        bnorm_mat = torch.mean(bulk_count, dim=1).view(-1, 1)
        self.bulk_count = bulk_count
        self.survival_time = survival_time
        self.cutting_off_0_1 = cutting_off_0_1
        self.bulk_norm_mat = bnorm_mat
        self.train_idx = torch.tensor([])
        self.validation_idx = torch.tensor([])
        self.test_idx = torch.tensor([])

    def bulk_split(self, bulk_seed, bulk_validation_num_or_ratio, bulk_test_num_or_ratio, censor_np, snv_flag=None, edges=None):
        if isinstance(censor_np, torch.Tensor):
            censor_np = censor_np.detach().cpu().numpy()
        if snv_flag is not None and isinstance(snv_flag, torch.Tensor):
            snv_flag = snv_flag.detach().cpu().numpy()
        if edges is None:
            raise ValueError("`edges` must be provided to bulk_split")
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
        edges = np.asarray(edges, float)

        surv_np = self.survival_time.detach().cpu().numpy()
        bin_idx = np.digitize(surv_np, edges, right=True) - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

        if snv_flag is None:
            strata = bin_idx * 2 + censor_np.astype(int)
        else:
            upper = bin_idx * 2 + censor_np.astype(int)
            n_bits = int(np.ceil(np.log2(snv_flag.max() + 1)))
            strata = (upper.astype(int) << n_bits) + snv_flag.astype(int)

        N = len(strata)
        val_ratio = (bulk_validation_num_or_ratio if bulk_validation_num_or_ratio < 1 else bulk_validation_num_or_ratio / N)
        test_ratio = (bulk_test_num_or_ratio if bulk_test_num_or_ratio < 1 else bulk_test_num_or_ratio / N)

        np.random.seed(bulk_seed)
        train_idx, val_idx, test_idx = [], [], []

        for k in np.unique(strata):
            idx_k = np.random.permutation(np.where(strata == k)[0])
            n_val = int(len(idx_k) * val_ratio)
            n_test = int(len(idx_k) * test_ratio)
            val_idx.extend(idx_k[:n_val])
            test_idx.extend(idx_k[n_val:n_val + n_test])
            train_idx.extend(idx_k[n_val + n_test:])

        for arr in (train_idx, val_idx, test_idx):
            np.random.shuffle(arr)

        self.train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        self.validation_idx = torch.as_tensor(val_idx, dtype=torch.long)
        self.test_idx = torch.as_tensor(test_idx, dtype=torch.long)
        self.time_bin_idx = torch.as_tensor(bin_idx, dtype=torch.long)
        
class SpatialDataManager():
    def __init__(self, spatial_count):
        spatial_count = spatial_count.float() 
        spatial_norm_mat = torch.mean(spatial_count, dim=1).view(-1, 1) 
        self.spatial_count = spatial_count
        self.spatial_norm_mat = spatial_norm_mat
