import torch
import numpy as np
import copy

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per_shot,n_per_query,args):
        self.n_batch = n_batch  
        self.n_cls = n_cls
        self.n_per = n_per_shot + n_per_query
        self.n_per_shot = n_per_shot
        self.n_per_query = n_per_query
        self.args = args

        label = np.array(label) 
        self.m_ind = []  
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.class_rng = np.random.RandomState(222)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        all_classes = list(range(self.args.meta_class))
        for i_batch in range(self.n_batch):
            support = []
            query = []

            classes = self.class_rng.choice(all_classes, size=5, replace=False)
            for key in classes:
                all_classes.remove(key)

            for c in classes:
                l = self.m_ind[c]  
                pos = torch.randperm(len(l))[:self.n_per]
                support.append(l[pos[:self.n_per_shot]])
                query.append(l[pos[self.n_per_shot:]])
            support_data = torch.stack(support,-1).t().reshape(-1)
            query_data = torch.stack(query,-1).t().reshape(-1)
            batch = torch.cat([support_data,query_data],-1)

            yield batch
