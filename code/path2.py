import os
path = '/etc'
filename = 'passwd'
if os.path.isdir(path):
    full_path = os.path.join(path,filename)
    if os.path.isfile(full_path):
        with open(full_path,'r') as f:
            line = f.readlines()
            for _ in range(len(line)):
                print(line)
