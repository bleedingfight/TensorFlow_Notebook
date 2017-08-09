import os
file_path = '~/iris_test.csv'
filename = os.path.basename(file_path)
new_dir = os.path.join('home','hpc',filename)
file_dir = os.path.dirname(file_path)
dir1 = '~/'
fulldir = os.path.expanduser(dir1)
sp = os.path.split(new_dir)
print('new_dir:',sp[0],'ext:',sp[1])

