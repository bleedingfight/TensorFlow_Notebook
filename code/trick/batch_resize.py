# TensorFlow1.3 GPU ubuntu14.0.4 64bit
# 文件和处理的图像放在一个目录
import tensorflow as tf
import cv2
from os import listdir
# 将图片压缩，默认压缩大小为[224,224]
def parse_function(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,3)
    image_resized = tf.image.resize_images(image_decoded,[224,224])
    return image_resized,label
# 过滤filepath路径中非ext格式的文件，默认过滤保留png格式图片
def filter_file(filepath,ext='png'):
    fileall = listdir(filepath)
    filted_list = []
    for i in fileall:
        if (len(i.split('.'))==2) and (i.split('.')[-1]=='png'):
            filted_list.append(i)
    return filted_list
# 压缩保存
def clip_pic(filenames,save_path):
    num_files = len(filenames)
    labels = tf.constant([0]*num_files)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(filenames),labels))
    dataset = dataset.map(parse_function)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    count =0
    for i in range(num_files):
        img = sess.run(next_element)[0]
        cv2.imwrite(save_path+filenames[i],img)
        count+=1
    print('成功压缩'+str(count)+'张图片')
def main():
    filepath = '/home/bleedingfight/Pylon/test/testma/'
    save_path = '/home/bleedingfight/test1/'
    filenames = filter_file(filepath)
    clip_pic(filenames,save_path)

if __name__ == '__main__':
    main()
