# Pascal VOC形式のxmlファイル群から,label_map.pbtxtを作成する．
# Object Detection APIにおいて，ラベルの数，ラベルの名称をを設定するためのファイル．

# ファイルは，以下の形式をとる．
## label_map.pbtxt 
# item {
#   id: 1
#   name: 'face'
# }

# item {
#   id: 2
#   name: 'maskface'
# }
# .
# .
# .
# ラベルマップの作成
import os
from lxml import etree
import tensorflow.compat.v1 as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('label_map_path', './label_map.pbtxt',
                    '(Relative) path to label map file outputed.')


FLAGS = flags.FLAGS

# generate label map from dataset in  Pascal VOC
def pascal_to_label_map(annotations_dir, label_map_path):
  # ファイル名のリストを取得
  annotations_list = os.listdir(annotations_dir)
  label_list=[]

  for index, annotation_filename in enumerate(annotations_list):
      # アノテーションデータをdictとして取得
      annotation_path = os.path.join(annotations_dir, annotation_filename)
      with tf.gfile.GFile(annotation_path, 'r') as fid:
          xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      
      # オブジェクト名のlistを作成
      for obj in data['object']:
          label = obj["name"]
          if label not in label_list:
              label_list.append(label)

  # generate label_map
  with open(label_map_path, 'w') as f:
    # Loop through all of the labels and write each label to the file with an id
    for idx, label in enumerate(label_list):
        f.write('item {\n')
        f.write("\tname: '{}'\n".format(label))
        f.write('\tid: {}\n'.format(idx + 1)) # indexes must start at 1
        f.write('}\n')
  print(label_map_path + " is created")

def main(_):
	AnotationsDir = FLAGS.annotations_dir
	LabelMapPath  = FLAGS.label_map_path

	pascal_to_label_map(AnotationsDir,LabelMapPath)



if __name__ == '__main__':
    tf.app.run()
    # これは何？
    # main関数を実行しているだけ．
    # tf.app.flagsでファイル実行時の標準入力の値を受領処理したあと，
    # main関数を実行する．
    # main関数は，引数を一つ受け取る必要があるが，多分ライブラリが利用するだけの実用的ではない引数．
    # 参考：https://neuryo.hatenablog.com/entry/2018/10/16/160110









