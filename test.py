import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import csv
import input

N_CLASSES = 12
CKPT_FILE = './inception_v3.ckpt'
LEARNING_RATE = 0.0001


def main():
    '''
    images = tf.placeholder(tf.float32,[None,299,299,3],name='input')
    labels = tf.placeholder(tf.int64,[None],name='labels')
    names = tf.placeholder(tf.string,[None],name='name')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits,_ = inception_v3.inception_v3(
            images,num_classes=N_CLASSES
        )


    loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels,N_CLASSES),logits,weights=1.0
    )

    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(
        tf.losses.get_total_loss()
    )
    with tf.name_scope('V_evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits,1),labels)
        evaluation_step = tf.reduce_mean(tf.cast(
            correct_prediction,tf.float32
        ))

    exclusions = ['InceptionV3/Logits',
                   'InceptionV3/AuxLogits']
    inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        inception_except_logits,
        ignore_missing_vars=True
    ) 
    '''

    names = tf.placeholder(tf.string,[None],name='name')
    images_feed,labels_feed,_ = input.get_batch("./training.tfrecords",15,299,299,2000,1999)
    images_E,labels_E,_ = input.get_batch("./validation.tfrecords",100,299,299,2000,1999)
    image_T,_,name_T = input.get_batch("./testing.tfrecords",1,299,299,2,1)

    saver = tf.train.import_meta_graph("./model/save_model.ckpt.meta")
    logits = tf.get_default_graph().get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
    images = tf.get_default_graph().get_tensor_by_name("input:0")
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        #images_feeds,name_feeds = sess.run([image_T,name_T])
        '''
        init = tf.global_variables_initializer()
        sess.run(init)

        L = sess.run(logits,feed_dict = {images:images_feeds})
        N = sess.run(names,feed_dict = {names:name_feeds})        
        print(L,N)
        '''
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        model_file=tf.train.latest_checkpoint('./model')
        saver.restore(sess,model_file)
        
        
        f = open('out_put.csv','w')
        f_csv = csv.writer(f)
        for i in range(440):
            '''
            images_feeds,name_feeds = sess.run([image_T,name_T])
            L = sess.run(tf.argmax(logits,1),feed_dict = {images:images_feeds})
            N = sess.run(names,feed_dict = {names:name_feeds})    
            '''
            images_feeds,name_feeds = sess.run([image_T,name_T]) 
            L = sess.run(logits,feed_dict = {images:images_feeds})
            N = sess.run(names,feed_dict = {names:name_feeds}) 
            print(N,L)
            #f_csv.writerows(L,N)
            
        coord.request_stop()
        coord.join(threads)   

if __name__ == '__main__':
    main()              
