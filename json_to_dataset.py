import argparse
import json
import os
import os.path as osp
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
import base64

import numpy as np
 
def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
 
    json_file = args.json_file
 
    count = os.listdir(json_file) 
    for i in range(0, len(count)):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path):
            data = json.load(open(path))

            ##########
            #save directory

            out_dir1 = osp.basename(path).split('.')[0]
            save_file_name = out_dir1
            out_dir1 = osp.join(osp.dirname(path), out_dir1)

            if not osp.exists(json_file + "//" + "labelme_json"):
                os.mkdir(json_file + "//" + "labelme_json")
            labelme_json = json_file + "//" + "labelme_json"

            out_dir2 = labelme_json + "//" + save_file_name
            if not osp.exists(out_dir2):
                os.mkdir(out_dir2)

            ##########
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
            lbl_viz = utils.draw_label(lbl, img, captions)
 
            PIL.Image.fromarray(img).save(osp.join(out_dir2, 'img.png'))
            utils.lblsave(osp.join(out_dir2, 'label.png'), lbl)
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir2, 'label_viz.png'))
 
            with open(osp.join(out_dir2, 'label_names.txt'), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')
 
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(out_dir2, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            # save img to another directory
            if not osp.exists(json_file + "img"):
                os.mkdir(json_file + "img")
            img_path = json_file + "img"
            PIL.Image.fromarray(img).save(osp.join(img_path, save_file_name + '.png'))

            # save png to another directory
            if not osp.exists(json_file + "mask"):
                os.mkdir(json_file + "mask")
            mask_save2png_path = json_file + "mask"

            utils.lblsave(osp.join(mask_save2png_path, save_file_name + ".png"), lbl)
 
            print('Saved to: %s' % out_dir2)

if __name__ == '__main__':
    main()

