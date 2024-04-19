
import numpy as np
import math



load_folder='Path_files_copy/path_4132/'



a = np.load(load_folder+'test_path0.npy')

b = np.load(load_folder+'test_path1.npy')
# b = np.delete(b, (0), axis=0)

c = np.load(load_folder+'test_path2.npy')
# c = np.delete(c, (0), axis=0)

d = np.load(load_folder+'test_path3.npy')
# d = np.delete(d, (0), axis=0)

e = np.load(load_folder+'test_path_final.npy')


print(d)
