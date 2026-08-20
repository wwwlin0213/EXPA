import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    obj_list = []  # Initialize obj_list
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'mvtec_ad_2':
        obj_list = ['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'dagm':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'ksdd':
        obj_list = ['electrical commutators']
    elif dataset_name == 'dtd':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name in ['oct17', 'OCT17', 'oct2017', 'Retina_OCT2017_AD']:
        obj_list = ['retina']
    elif dataset_name in ['brainmr1', 'BrainMR1', 'brainmri', 'BrainMRI']:
        obj_list = ['brain']
    elif dataset_name in ['brain_ad', 'Brain_AD', 'brainad', 'BrainAD']:
        obj_list = ['brain']
    elif dataset_name in ['his', 'HIS', 'histopathology_ad', 'Histopathology_AD']:
        obj_list = ['histopathology']
    elif dataset_name in ['cvc', 'CVC', 'cvc_clinicdb', 'CVC-ClinicDB', 'clinicdb']:
        obj_list = ['cvc_clinicdb']
    elif dataset_name in [
        'endo',
        'endotect',
        'EndoTect',
        'EndoTect_2020_Segmentation_Test_Dataset',
        'EndoTect_2020'
    ]:
        obj_list = ['endo']
    elif dataset_name in ['kvasir', 'Kvasir', 'kvasir_seg', 'Kvasir-SEG']:
        obj_list = ['kvasir']
    elif dataset_name == 'headct':
        obj_list = ['brain']
    elif dataset_name == 'br35h':
        obj_list = ['brain']
    elif dataset_name == 'Chest':
        obj_list = ['chest']

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        # meta_info = meta_info[mode]

        # === 修改开始 ===
        # 兼容 meta.json 中的键名为 'Test' 或 'test' 的情况
        mode_key = mode.capitalize()   # 'test' -> 'Test', 'train' -> 'Train'
        if mode_key not in meta_info and mode in meta_info:
            mode_key = mode
        meta_info = meta_info[mode_key]
        # === 修改结束 ===

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        # Handle unknown class names by assigning a default ID
        cls_id = self.class_name_map_class_id.get(cls_name, 0)
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": cls_id}    