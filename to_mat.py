import scipy.io as scio
import numpy as np

data = scio.loadmat('D:\\wwg\\nuswide.mat')
codes = data['r_img']
labels = data['r_l']

# find signal label featrue
for i in range(21):  # NUSWIDE-TC21 has 21 class
    print('i-----',i)
    o = np.zeros((21,))
    o[i] = 1
    tot = 0
    wanted_code = []
    for k, j in enumerate(labels):
        if np.array_equal(o, j):  # judge that two vector is completely equal
            tot = tot + 1
            wanted_code.append(codes[k])

    print(tot)
    wanted_code = np.array(wanted_code)
    save_label_dist = {
        'hash': wanted_code
    }
    print(wanted_code.shape)
    # print(wanted_label[:10])
    scio.savemat(f'./to_mat/{i}_{wanted_code.shape[0]}.mat', mdict=save_label_dist)