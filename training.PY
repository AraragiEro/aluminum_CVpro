#第四个PY文件，加载处理好的数据和模型，并训练

#加载模型
import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import tensorflow.contrib.slim as slim
import input
import csv
import pandas as pd
import numpy as np

#加载处理好的数据
TRAINING_FILE = './training.tfrecords'  #输出文件
TESTING_FILE = './testing.tfrecords'
VALIDATION_FILE = './validation.tfrecords'

#保存训练好的模型路径
TRAIN_FILE = './model/save_model.ckpt'
write_csv = "train.csv"

#加载谷歌提供的模型数据
CKPT_FILE = './inception_v3.ckpt'
data_counts = [1018,39,139,47,90,248,77,54,43,35,209,140]

#训练参数
LEARNING_RATE = 0.0001
STEPS = 1800
BATCH = 32
N_CLASSES = 12   #???? 
Training_batch = 5

#全连接层参数前缀
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

#从谷歌模型中加载参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded = True
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = False
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

#获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    #枚举所有需要训练的变量前缀，找到所有参数
    for scope in scopes:
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variables_to_train.extend(variables)
    return variables_to_train

def get_loss_bias():
    bias = [[data_counts[j]/data_counts[i] for j in range(N_CLASSES)] for i in range(N_CLASSES)]
    return bias

def get_loss_weight(batch_size,labels_size,labels,weight_array):
    weight = [weight_array[labels[i]] for i in range(batch_size)]
    return weight

def main():
    tf.reset_default_graph()
    #加载与处理好的数据
    images = tf.placeholder(tf.float32,[None,299,299,3],name='input')
    labels = tf.placeholder(tf.int64,[None],name='labels')
    names = tf.placeholder(tf.string,[None],name='name')
    loss_weights = tf.placeholder(tf.float32,[None,12],name='loss_weight')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits,_ = inception_v3.inception_v3(
            images,num_classes=N_CLASSES
        )
    trainable_variables = get_trainable_variables()
    '''
    tf.losses.softmax_cross_entropy(
        tf.one_hot(labels,N_CLASSES),logits,weights=1.0
    )
    '''
    with tf.name_scope('losses'):
        loss = tf.reduce_mean((tf.one_hot(labels,N_CLASSES) - logits)**2 * loss_weights)#
        tf.losses.add_loss(loss)
        tf.summary.scalar('losses',tf.losses.get_total_loss())
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
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

    images_feed,labels_feed,_ = input.get_batch("./training.tfrecords",Training_batch,299,299,2000,1999)
    images_E,labels_E,_ = input.get_batch("./validation.tfrecords",100,299,299,2000,1999)
    image_T,labels_T,name_T = input.get_batch("./testing.tfrecords",10,299,299,2000,1,if_shuffer=False)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        '''
        log_writer = tf.summary.FileWriter("./log/",sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        print('建立输入队列')
        init = tf.global_variables_initializer()
        sess.run(init)
        load_fn(sess)
        print('载入模型数据')
        IF_TRAIN = False
        print('训练开始')
        for i in range(STEPS):
            images_feeds,labels_feeds = sess.run([images_feed,labels_feed])
            weights_feed = get_loss_weight(Training_batch,N_CLASSES,labels_feeds,get_loss_bias())
            #print(np.array(weights_feed).shape)
            
            sess.run(train_step,feed_dict={images:images_feeds,
                                            labels:labels_feeds,
                                            loss_weights:weights_feed})
            summary = sess.run(merged,feed_dict={images:images_feeds,
                                            labels:labels_feeds,
                                            loss_weights:weights_feed})
            print('第 %d 轮完成,损失为：' % i,sess.run(tf.losses.get_total_loss(),feed_dict={images:images_feeds,
                                                                                labels:labels_feeds,
                                                                                loss_weights:weights_feed}))
            if i % 30 ==0 or i + 1 == STEPS:        #输出日志
                saver.save(sess,"./model/save_model.ckpt")
                images_Es,labels_Es = sess.run([images_E,labels_E])
                validation_accuracy = sess.run(evaluation_step,feed_dict={images:images_Es,
                                                                            labels:labels_Es
                })
                print('第 %d 论训练后，正确率估计：%.1f%%' % (i,validation_accuracy * 100))
                print(sess.run(tf.arg_max(logits,1),feed_dict={images:images_Es,labels:labels_Es}))
                #print(sess.run(labels,feed_dict={images:images_Es,labels:labels_Es}))
            log_writer.add_summary(summary,i)       
        coord.request_stop()
        coord.join(threads)  
        log_writer.close()
        '''
        print('建立输入队列')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        print('载入已训练模型数据')
        model_file=tf.train.latest_checkpoint('./model')
        saver.restore(sess,model_file)

        head = True
        data_bool = [1 for x in range(440)]
        print('分类开始')
        for i in range(100):
            print('第%d组' % i)
            images_Es,labels_Es,names_ES = sess.run([image_T,labels_T,name_T])
            #print('第 %d 论训练后，正确率估计：%.1f%%' % (0,validation_accuracy * 100))
            L = sess.run(tf.argmax(logits,1),feed_dict={images:images_Es})
            N = sess.run(names,feed_dict={names:names_ES})
            for j in range(10):
                not_finish = 1 in data_bool
                if not not_finish:
                    break
                numb = str(N[j])[2:-5]
                numb = int(numb)
                if data_bool[numb] == 1:
                    if 'defect'+str(L[j]) == 'defect0':
                        L_b = 'norm'
                    else:
                        L_b = 'defect'+str(L[j])
                    print('第%d个' % j)
                    print(str(N[j])[2:-1],L_b)
                    df = pd.DataFrame(({'defect1':[L_b]}),index = [str(N[j])[2:-1]])
                    if head:
                        head = False
                        df.to_csv('out_put.csv', mode='w', header=None)
                    else:
                        df.to_csv('out_put.csv', mode='a', header=None)
                    data_bool[numb] = 0
            if not not_finish:
                break
        
        coord.request_stop()
        coord.join(threads)  
        

if __name__ == '__main__':
    main()
        
