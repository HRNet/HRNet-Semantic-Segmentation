import glob
import os
import os.path as osp

basedir = '/mnt/openseg/data/cityscapes/'

lists = {'train': [], 'val': []}

for split in ['train', 'val']:
    for fn in os.listdir(osp.join(basedir, split, 'image')):
        lists[split].append([
            osp.join(split, 'image', fn),
            osp.join(split, 'label', fn.rpartition('.')[0] + '.png'),
        ])

lists['trainval'] = sum(lists.values(), [])
for key, lst in lists.items():
    with open(osp.join(key+'.lst'), 'w') as f:
        for img, label in lst:
            f.write('{} {}\n'.format(img, label))