# 为了修改文件名称，去掉时间后缀的程序文件

import os

path = "D:\\workspace\\QuantumBase\\ImportantData\\RandomCircuitOnCloud"

file_name_list = os.listdir(path)

for file_name in file_name_list:
    if file_name[0] != 'R':
        new_name = file_name[0:file_name.index('_')] + '.xlsx'
        os.rename(path + '\\' + file_name, path + '\\' + new_name)
