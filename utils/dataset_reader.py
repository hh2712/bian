import torch
from utils import DGraphFin


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
    else:
        raise ValueError("Unknown dataset")
    return data

