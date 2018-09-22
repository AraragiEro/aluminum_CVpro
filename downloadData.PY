#思路复刻

#第一个PY文件，用于下载图像数据库
import wget
import tarfile
#第一步，下载
URL="http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
OUT_FNAME = "inception_v3_2016_8_28.tar.gz"
wget.download(URL,out=OUT_FNAME)
#第二部，解压
tar = tarfile.open(OUT_FNAME)
tar.extractall()
tar.close()
