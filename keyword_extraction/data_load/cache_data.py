from load import load_data
import csv

if __name__ == "__main__":
    texts = load_data('../data/train2')

    with open('../cache/train.csv', encoding='utf8', mode="w", newline='') as f:
        fieldnames = ['texts', 'process_tags', 'material_tags', 'task_tags']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for text in texts:
            writer.writerow(text)

    texts = load_data('../data/dev')

    with open('../cache/dev.csv', encoding='utf8', mode="w", newline='') as f:
        fieldnames = ['texts', 'process_tags', 'material_tags', 'task_tags']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for text in texts:
            writer.writerow(text)
