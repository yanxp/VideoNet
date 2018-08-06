import collections
d = collections.defaultdict(list)
fs = open('data/meitu_splits/classind.txt','r')
classind = [line.strip().split(',') for line in fs.readlines()]
for i in range(len(classind)):
    d[i] = classind[i][1]
