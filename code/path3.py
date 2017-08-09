import os
import os.path
import glob
import time
path_name = '/home/hpc/TensorFlow_Notebook/code'
pyfiles = glob.glob(path_name+'/*.py')
name_sz_data = [(name,os.path.getsize(name),os.path.getmtime(name)) for name in pyfiles]
file_metadata = [(name,os.stat(name)) for name in pyfiles]
for name,meta in file_metadata:
    print(name,'\t|',meta.st_size,'\t|',time.ctime(meta.st_mtime))

