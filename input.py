import tensorflow as tf
import matplotlib.pyplot as plt

def get_batch(file_name,batch_size,x_size,y_size,capacity=100,min_after_dequeue=100,if_shuffer = True):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([file_name])
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'base_name':tf.FixedLenFeature([],tf.string)
        })
    file_list = features['image_raw']
    label = tf.cast(features['label'],tf.int64)
    name = features['base_name']
    image_raw = tf.read_file(file_list)
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image,[x_size,y_size],method=1)
    image = tf.reshape(image,[x_size,y_size,3])
    if if_shuffer == True:
        image_batch,label_batch,name_batch = tf.train.shuffle_batch([image,label,name],
                                                    batch_size=batch_size,
                                                    num_threads=64,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
    else:
        image_batch,label_batch,name_batch = tf.train.batch([image,label,name],
                                                    batch_size=batch_size,
                                                    num_threads=64,
                                                    capacity=capacity)
        #image_batch = tf.cast(image_batch,tf.uint8)
    return image_batch,label_batch,name_batch

def main():    
    image,label,name = get_batch("./testing.tfrecords",100,299,299,2000,1999)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess= sess,coord=coord)
    images,labels,names = sess.run([image,label,name])
    print(names[45])
    plt.imshow(images[45])
    plt.show()
    '''
    file_list = features['image_raw']
    print('文件路径',features['image_raw'])
    label = tf.cast(features['label'],tf.int64)
    print('label',[label])
    image_raw = tf.read_file(file_list)
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image,[299,299])
    image_batch,label_batch = tf.train.batch([image,label],
                                                batch_size=32,
                                                num_threads=64,
                                                capacity=100)
                                                '''

if __name__ == '__main__':
    main()