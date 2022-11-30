import torch
from utils import DGraphFin, Ethereum, EthereumSO
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T


def read_dataset(dataset_name, root_dir = "/data/huhy/datasets/", transform=None, **kwargs):
    if dataset_name == "DGraphFin":
        dataset = DGraphFin(root_dir, name=dataset_name, transform=transform)
        data = dataset[0]
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
        data.edge_attr = data.edge_attr.unsqueeze(1)
        data.n_id = torch.arange(data.x.size(0))
        # data.e_id = torch.arange(data.edge_index.size(1))
        data.train_nodes = data.train_mask
        data.valid_nodes = data.valid_mask
        data.test_nodes = data.test_mask
        data.train_mask = torch.isin(data.n_id, data.train_mask)
        data.valid_mask = torch.isin(data.n_id, data.valid_mask)
        data.test_mask = torch.isin(data.n_id, data.test_mask)
        data.reminder_mask = torch.bitwise_not(torch.isin(data.n_id,
                                                          torch.cat(
                                                              [data.train_nodes, data.valid_nodes, data.test_nodes])))
        data.reminder_nodes = data.n_id[data.reminder_mask]

    elif dataset_name == "ethereum":
        dataset = Ethereum(root_dir)
        data = dataset[0]
        data.y = data.y.unsqueeze(1)
        if kwargs.get("coalesce") and kwargs.get("attr") == "edge_timestamp":
            edge_index, edge_timestamp = coalesce(data.edge_index, data.edge_timestamp, reduce=kwargs.get("reduce", ",mean"))
            data.edge_index = edge_index
            data.edge_timestamp = edge_timestamp
            del data.edge_attr
        elif kwargs.get("coalesce") and kwargs.get("attr") == "edge_attr":
            edge_index, edge_attr = coalesce(data.edge_index, data.edge_attr, reduce=kwargs.get("reduce", ",mean"))
            data.edge_index = edge_index
            data.edge_attr = edge_attr.unsqueeze(1)
            del data.edge_timestamp
        if transform:
            transform = T.ToSparseTensor()
            data = transform(data)
    elif dataset_name == "ethereumso":
        dataset = EthereumSO(root_dir)
        data = dataset[0]
        data.y = data.y.unsqueeze(1)
        data.edge_timestamp = data.edge_timestamp[:, 0]
        if kwargs.get("coalesce") and kwargs.get("attr") == "edge_timestamp":
            edge_index, edge_timestamp = coalesce(data.edge_index, data.edge_timestamp,
                                                  reduce=kwargs.get("reduce", ",mean"))
            data.edge_index = edge_index
            data.edge_timestamp = edge_timestamp
            del data.edge_attr
        elif kwargs.get("coalesce") and kwargs.get("attr") == "edge_attr":
            edge_index, edge_attr = coalesce(data.edge_index, data.edge_attr, reduce=kwargs.get("reduce", ",mean"))
            data.edge_index = edge_index
            data.edge_attr = edge_attr.unsqueeze(1)
            del data.edge_timestamp
        if transform:
            transform = T.ToSparseTensor()
            data = transform(data)
    elif dataset_name == "ogbn-proteins":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=dataset_name, root=root_dir)
        data = dataset[0]
        data.x = data.node_species
        data.n_id = torch.arange(data.x.size(0))
        data_split = dataset.get_idx_split()
        data.train_nodes = data_split["train"]
        data.valid_nodes = data_split["valid"]
        data.test_nodes = data_split["test"]
        data.train_mask = torch.isin(data.n_id, data.train_nodes)
        data.valid_mask = torch.isin(data.n_id, data.valid_nodes)
        data.test_mask = torch.isin(data.n_id, data.test_nodes)

    else:
        raise ValueError("Unknown dataset")
    return data

