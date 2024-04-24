
import numpy as np
import math



load_folder='Path_files_copy/path_2134/'



# a = np.load(load_folder+'test_path0.npy')

# b = np.load(load_folder+'test_path1.npy')
# # b = np.delete(b, (0), axis=0)

# c = np.load(load_folder+'test_path2.npy')
# # c = np.delete(c, (0), axis=0)

# d = np.load(load_folder+'test_path3.npy')
# # d = np.delete(d, (0), axis=0)

# e = np.load(load_folder+'test_path_final.npy')


# print(a)

order=[3,3,1,2,3]
load_folder=""
path=[]
for o in order:
    load_folder+=f"{o}"
load_folder="Path_files/path_"+load_folder + "/"

for i,o in enumerate(order):
    path.append(np.load(load_folder+f'test_path{i}.npy'))
path.append(np.load(load_folder+'test_path_final.npy'))


for i,value in enumerate(path):
    print("path for",i,value )

