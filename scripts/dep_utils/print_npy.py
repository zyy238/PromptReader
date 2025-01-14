import numpy as np

#Import npy file path location
test = np.load('all_emb.npy')
# test = np.load('emb.npy')
print('len(test)=', len(test))
print(len(test[0]))
print(test)

