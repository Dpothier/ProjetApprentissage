import torch
from collections import Counter

def get_tags_weight_ratio(examples):
    all_process = [tag for example in examples for tag in example['process_tags']]
    all_material = [tag for example in examples for tag in example['material_tags']]
    all_task = [tag for example in examples for tag in example['task_tags']]

    all_tags = all_process + all_material + all_task

    total_number_of_tags = len(all_tags)
    tags_count = Counter(all_tags)
    tags_count_list = [tags_count[0], tags_count[1]]

    print("1 counts of {}% of data".format(tags_count[1]/total_number_of_tags * 100))

    return torch.Tensor(tags_count_list)