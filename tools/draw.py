import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Cifar100 classification Acc@top1 curve tool')
parser.add_argument('csv_dir', metavar='DIR', help='path to dataset')
args = parser.parse_args()

# `csv_dir` download from VisualDL scalar
# csv_dir = 'visualdl-scalar-eval_acc_top1/'
plt.figure(figsize=(15, 10))
for file_name in sorted(os.listdir(args.csv_dir)):
    if not file_name.startswith('visualdl'):
        continue
    label_name = file_name.split('-')[2].lstrip('logs_')
    path = os.path.join(args.csv_dir, file_name)
    x = []
    y = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            _items = line.strip().split(',')
            x.append(int(_items[0]))
            y.append(float(_items[-1]))
    plt.plot(x, y, lw=1, c=np.random.uniform(size=3), label=label_name)
plt.title('Cifar100 classification Acc@top1')
plt.xlabel('epoch')
plt.ylabel('Acc@top1')
plt.legend(loc='upper left')
plt.savefig('.github/acc_top1_curve.png')
