import numpy as np

#导入npy文件路径位置
test = np.load('all_emb.npy')
# test = np.load('emb.npy')
print('len(test)=', len(test))
print(len(test[0]))
print(test)

