import tensorflow as tf
import os
import glob
import numpy as np

INPUT_DATA = './TRAINING_DATA'
TEST_DATA = './TESTING_DATA'
TRAINING_FILE = './training.tfrecords'
VALIDATION_FILE = './validation.tfrecords'
TESTING_FILE = './testing.tfrecords'

validation_procentage = 10

def creat_training_data(sess ,validation_procentage):
    writer_training = tf.python_io.TFRecordWriter(TRAINING_FILE)
    writer_validation = tf.python_io.TFRecordWriter(VALIDATION_FILE)

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #print(sub_dir)
    #print(sub_dir[1][16:18])
    is_root_dir = True                              #即5个种类的花。
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg','jpgs']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        current_label = int(dir_name)
        #print(sub_dir,':',dir_name,':',current_label)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
            else:
                print('正在分类文件夹：',sub_dir)
                #print(file_list[5],file_list[5].encode())
            
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            base_name_code = base_name.encode()
            file_name_code = file_name.encode()
            chance = np.random.randint(100)
            if chance < validation_procentage:
                example = tf.train.Example(features = tf.train.Features(feature = {
                'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name_code])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
                'base_name':tf.train.Feature(bytes_list = tf.train.BytesList(value = [base_name_code]))
                }))
                writer_validation.write(example.SerializeToString())
            else:
                example = tf.train.Example(features = tf.train.Features(feature = {
                'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name_code])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
                'base_name':tf.train.Feature(bytes_list = tf.train.BytesList(value = [base_name_code]))

                }))
                writer_training.write(example.SerializeToString())
            print('正在分类文件：',file_name,'label:',current_label)
    writer_training.close()
    writer_validation.close()
    print('分类完成')

def creat_test_data(sess):
    current_label = 13
    writer_testing = tf.python_io.TFRecordWriter(TESTING_FILE)
    extensions = ['jpg','jpgs']
    file_list = []  
    for extension in extensions:
        file_glob = os.path.join(TEST_DATA,'*.'+extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        else:
            print('正在整理测试数据')
            #print(file_list[5],file_list[5].encode())
    for file_name in file_list:
        
        base_name = os.path.basename(file_name)
        base_name_code = base_name.encode()
        #print('base_name:',base_name)
        file_name_code = file_name.encode()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name_code])),
            'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
            'base_name':tf.train.Feature(bytes_list = tf.train.BytesList(value = [base_name_code]))
        }))
        writer_testing.write(example.SerializeToString())
        print('正在分类：',file_name,'    ',base_name)
    writer_testing.close()
    print('分类完成')
    



def main():
    with tf.Session() as sess:
        
        #creat_training_data(sess,validation_procentage)
        creat_test_data(sess)

if __name__ == '__main__':
    main()

