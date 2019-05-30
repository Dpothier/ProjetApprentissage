import torch.utils.data as data

class DataLoader(data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = dataset.get_collate_fn()
        super(DataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn)

