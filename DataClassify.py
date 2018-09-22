#第二个PY文件，分类下载的图片并保存为NP文件供tensorflow使用

import os
import glob
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from PIL import Image

INPUT_DATA = './flower_photos'          #输入路径
TRAINING_FILE = './training.tfrecords'  #输出文件
TESTING_FILE = './testing.tfrecords'
VALIDATION_FILE = './validation.tfrecords'

TEST_PROCENTAGE = 10
VALIDATION_PROCENTAGE = 10

#读取 并分割数据的函数
def create_image_lists(sess ,testing_percentage ,validation_percentage):
    writer_training = tf.python_io.TFRecordWriter(TRAINING_FILE)
    writer_testing = tf.python_io.TFRecordWriter(TESTING_FILE)
    writer_validation = tf.python_io.TFRecordWriter(VALIDATION_FILE)

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  #遍历，并输出该目录下所有文件夹名到sub_dirs。
    #print('sub',sub_dirs)
    is_root_dir = True                              #即5个种类的花。


    #定义训练，验证，测试集以及对应label

    current_label = 0

    #遍历每个文件夹
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        #获取该文件架下所有图片文件
        extensions = ['jpg','jpeg']
        file_list = []
        dir_name = os.path.basename(sub_dir)    #返回该文件夹下文件列表的最后一个文件
        for extension in extensions:    #遍历所有文件类型
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))      #路径符合↑条件的将路径加入file_list组
            #print('\n',extension,'\n',file_glob,'\n',glob.glob(file_glob))
            print('正在分类文件夹：',sub_dir)
            if not file_list:
                continue
            
            #处理获取后的文件           此处有点小问题，如果文件后缀不统一是否会产生重复？
        for file_name in file_list:

            #image_raw_data = gfile.FastGFile(file_name,'rb').read() #返回??????
            #image = tf.image.decode_jpeg(image_raw_data)            #返回unit-8张量
            #image = tf.image.resize_images(image,[299,299])
            #image = tf.image.convert_image_dtype(image ,dtype = tf.uint8)

            '''image_raw_data = Image.open(file_name)
            image_raw_data = image_raw_data.resize((299,299))
            image_raw = image_raw_data.tobytes()'''
            file_name = file_name.encode()
            #image_value = sess.run(image)
            
            #根据开始定义的概率随机分配数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                #validation_images.append(image)
                #validation_labels.append(current_label)
                example = tf.train.Example(features = tf.train.Features(feature = {
                'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
                }))
                writer_validation.write(example.SerializeToString())
            elif chance < (testing_percentage + validation_percentage):
                #testing_images.append(image)
                #testing_labels.append(current_label)
                example = tf.train.Example(features = tf.train.Features(feature = {
                'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
                }))
                writer_testing.write(example.SerializeToString())
            else:
                #training_images.append(image)
                #training_labels.append(current_label)
                example = tf.train.Example(features = tf.train.Features(feature = {
                'image_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name])),
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [current_label])),
                }))
                writer_training.write(example.SerializeToString())
            print('正在分类文件：',file_name,'label:',current_label)
        current_label += 1
    writer_testing.close()
    writer_training.close()
    writer_validation.close()
    print('分类完成2')
    
        
        
    
    #将数据随机打乱
    #state = np.random.get_state()
    #np.random.shuffle(training_images)
    #np.random.set_state(state)
    #np.random.shuffle(training_labels)
    
    #return np.asarray([training_images,training_labels,validation_images,validation_labels,testing_images,testing_labels])


def main():
    sess = tf.Session()
        #process_data = create_image_lists(
        #    sess,
        #    TEST_PROCENTAGE,
        #    VALIDATION_PROCENTAGE,
        #)
    sess.run(create_image_lists(
        sess,
        TEST_PROCENTAGE,
        VALIDATION_PROCENTAGE,
    ))
    sess.run(print('分类完成1'))
    sess.close()
        #np.save(OUTPUT_FILE,process_data)
        

if __name__ =='__main__':
    main() 

    

