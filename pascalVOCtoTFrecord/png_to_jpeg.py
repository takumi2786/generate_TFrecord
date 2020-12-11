# pascal形式のデータで，png画像を含むデータセットを，jpegに変換する．
import os, glob
from PIL import Image
import tensorflow.compat.v1 as tf
import pathlib
import xml.etree.ElementTree as ET 

flags = tf.app.flags
flags.DEFINE_string('image_input', 'image_png',
                    'path to png images')
flags.DEFINE_string('image_out', 'image',
                    'path to jpg images outputed')
flags.DEFINE_string('xml_input', 'image_png',
                    'path to original xml annotations')
flags.DEFINE_string('xml_out', 'image_png',
                    'path to xml annotations outputed')

FLAGS = flags.FLAGS



# png画像をjpeg画像に変換
def convert_images_png_to_jpeg(input_path, out_path,flag_delete_original_files=False):
    filepath_list = glob.glob(input_path + '/*.png') # .pngファイルをリストで取得する
    for filepath in filepath_list:
        basename  = os.path.basename(filepath) # ファイルパスからファイル名を取得
        save_filepath = out_path + '/' + basename [:-4] + '.jpg' # 保存ファイルパスを作成
        img = Image.open(filepath)
        img = img.convert('RGB') # RGBA(png)→RGB(jpg)へ変換
        img.save(save_filepath, "JPEG", quality=95)
        print(filepath, '->', save_filepath)
        if flag_delete_original_files:
            os.remove(filepath)
            print('delete', filepath)
    return

# アノテーションファイルのファイル名拡張子を.pngから.jpgに変換
def convert_xmls_png_to_jpeg(input_path, out_path):
    filepath_list = glob.glob(input_path + '/*')
    for annotation_path in filepath_list:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for element in root.iter('filename'):
            img_file_name = element.text
            without_ext = pathlib.PurePath( img_file_name).stem
            element.text=without_ext+".jpg"#要素を書き換え．
    
        xml_file_name = os.path.basename(annotation_path)# xmlファイル名
        tree.write(out_path+'/'+xml_file_name, encoding='UTF-8')

    return

def main(_):
    image_input=FLAGS.image_input
    image_out=FLAGS.image_out
    xml_input=FLAGS.xml_input
    xml_out=FLAGS.xml_out
    convert_images_png_to_jpeg(image_input,image_out)
    convert_xmls_png_to_jpeg(xml_input,xml_out)



if __name__ == "__main__":
    tf.app.run()