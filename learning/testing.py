from getData import get_data_from_both_datasets


bdrv_data, bdrv_targets, carcomplaints_data, carcomplaints_targets = get_data_from_both_datasets()

shared_targets = ['steering', 'brakes', 'electrical', 'engine', 'suspension']

print(set(bdrv_targets))
print(set(carcomplaints_targets))
print(len(bdrv_data))
print(len(carcomplaints_data))

