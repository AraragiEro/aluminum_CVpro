import tensorflow as tf
import os
import glob
import pandas as pd
import numpy as np

validation_procentage = 10
INPUT_DATA = './TRAINING_DATA_VOL2'
TEST_DATA = './TESTING_DATA'
TRAINING_DATA = './training_data.csv'
VALIDATION_DATA = './validation_data.csv'
DATA = './data.csv'

dic = { '正常':'norm',
        '不导电':'defect1',
        '擦花':'defect2',
        '横条压凹':'defect3',
        '桔皮':'defect4',
        '漏底':'defect5',
        '碰伤':'defect6',
        '起坑':'defect7',
        '凸粉':'defect8',
        '涂层开裂':'defect9',
        '脏点':'defect10',
        '其他':'defect11',
        '变形':'defect11',
        '驳口':'defect11',
        '打白点':'defect11',
        '打磨印':'defect11',
        '返底':'defect11',
        '火山口':'defect11',
        '角位漏底':'defect11',
        '铝屑':'defect11',
        '喷流':'defect11',
        '喷涂碰伤':'defect11',
        '碰凹':'defect11',
        '漆泡':'defect11',
        '气泡':'defect11',
        '伤口':'defect11',
        '拖烂':'defect11',
        '纹粗':'defect11',
        '油印':'defect11',
        '油渣':'defect11',
        '杂色':'defect11',
        '粘接':'defect11',
        '划伤':'defect11'}

def creat_csv():
    x=0
    sub_dirs = [x[0]for x in os.walk(INPUT_DATA)]
    #print(sub_dirs)
    is_root_dir = True
    df = pd.DataFrame({'file_name':1,'defect_name':2,'label':3,'dir':4},
                        columns=['file_name','defect_name','label','dir'],index=[])
    df.to_csv(DATA,mode='w',header=['file_name','defect_name','label','dir'])

    df = pd.DataFrame({'file_name':1,'defect_name':2,'label':3,'dir':4},
                        columns=['file_name','defect_name','label','dir'],index=[])
    df.to_csv(VALIDATION_DATA,mode='w',header=['file_name','defect_name','label','dir'])
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        
        extensions = ['jpg','jpgs']
        file_list = []
        dir_name = os.path.basename(sub_dir)    #该文件夹下所有文件的文件名
        defect = sub_dir.split("\\")[-1]        #分割获得瑕疵名
        defect_label = dic[defect]              #查找字典获得瑕疵编号
        if defect_label == 'norm':              #得到下次label
            label = 0
        else:
            label = defect_label[6:]
        print(label)
        
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
            else:
                print('正在分类文件夹：',sub_dir)
                #print(file_list[5],file_list[5].encode())
        for file_name in file_list:
            x=x+1
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            
            if chance < validation_procentage:
                df = pd.DataFrame({'file_name':base_name,'defect_name':defect,'label':defect_label,'dir':file_name},
                        columns=['file_name','defect_name','label','dir'],index=[' '])
                df.to_csv(DATA,mode='a',header=None)
            else:
                df = pd.DataFrame({'file_name':base_name,'defect_name':defect,'label':defect_label,'dir':file_name},
                        columns=['file_name','defect_name','label','dir'],index=[' '])
                df.to_csv(DATA,mode='a',header=None)
                
            if defect_label == 'defect11':
                print(file_name,'分类完成',defect_label)
    print(x)

def main():
    creat_csv()

if __name__ == '__main__':
    main()