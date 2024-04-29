#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp
import numpy as np
from preprocess import *

class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))]) # [-1,1]
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                if opt.pair_setting == 'pair':
                    c_name = im_name.replace("_0", "_1")
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # cloth
        cloth = np.array(Image.open(osp.join(self.data_path, 'cloth', c_name))).astype(np.float32)/255
        cloth_tenor = torch.from_numpy(cloth.transpose(2,0,1))
        # mloth
        mloth = np.array(Image.open(osp.join(self.data_path, 'cloth-mask', c_name))).astype(np.float32)/255
        mloth_tensor = torch.from_numpy(mloth).unsqueeze(0)
        # parse
        parse = np.array(Image.open(osp.join(self.data_path, 'image-parse', im_name.replace('.jpg', '.png')))).astype('uint8')
        parse = ParseFine(parse) # [0-19] -> [0-6]
        # image
        image_array = np.array(Image.open(osp.join(self.data_path, 'image', im_name))).astype(np.float32)/255
        image = torch.from_numpy(image_array.transpose(2,0,1))
        # mask
        img_cloth_high, mask, parse_mask = mask_parse(parse)
        parse_mask = parse_mask.astype('uint8')
        # parse7_occ
        parse7_s = ParseOneHot(parse, num_class=7)
        parse7_s = torch.from_numpy(parse7_s.astype(np.float32))
        parse7_occ = ParseOneHot(parse_mask, num_class=7).astype(np.float32)
        # hand
        hand_mask_arr = np.array(Image.open(osp.join(self.data_path, 'other', 'hand', im_name.replace("jpg", "png"))).convert("L")).astype(np.float32)
        hand_mask_arr = np.expand_dims(np.clip(hand_mask_arr,0,1), axis=2)
        # image_occ
        image_occ_arr = mask_image(mask, image.numpy().transpose(1,2,0))
        image_occ_arr = image_occ_arr * (1-hand_mask_arr) + image_array * hand_mask_arr
        image_occ = torch.from_numpy(image_occ_arr.transpose(2,0,1).astype(np.float32))
        # pose_map18
        pose_map18 = get_pose_map18(im_name, self.data_path, self.fine_height, self.fine_width, 5, self.transform)

        # tensor - [0,1]
        result = {
            'c_name':               c_name,                 # list
            'im_name':              im_name,                # list

            # input
            'cloth':                cloth_tenor,
            'mloth':                mloth_tensor,           # [b, 3, 256, 192]
            'pose_map18':           pose_map18,             # [b, 18, 256, 192]
            'parse7_occ':           parse7_occ,             # [b, 7, 256, 192]
            'image_occ':            image_occ,              # [b, 3, 256, 192]

            'image':                image,                  # [b, 3, 256, 192] 
            'parse':                parse,                  # [b, 1, 256, 192] 
            }          

        return result

    def __len__(self):
        return len(self.im_names)

class Dataset2(data.Dataset):
    def __init__(self, opt):
        super(Dataset2, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.mid_data = opt.mid_data
        self.datamode = opt.datamode
        self.pair_setting = opt.pair_setting
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))]) # [-1,1]
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                if opt.pair_setting == 'pair':
                    c_name = im_name.replace("_0", "_1")
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # cloth
        cloth = np.array(Image.open(osp.join(self.mid_data, self.datamode, self.pair_setting, 'warp-cloth', c_name))).astype(np.float32)/255
        cloth_tenor = torch.from_numpy(cloth.transpose(2,0,1))
        # mloth
        warp_mloth_np = np.array(Image.open(osp.join(self.mid_data, self.datamode, self.pair_setting, 'warp-mloth', c_name))).astype(np.float32)/255
        mloth_tensor = torch.from_numpy(warp_mloth_np).unsqueeze(0)
        # parse
        parse = np.array(Image.open(osp.join(self.data_path, 'image-parse', im_name.replace('.jpg', '.png')))).astype('uint8')
        parse = ParseFine(parse) # [0-19] -> [0-6]
        # image
        image_array = np.array(Image.open(osp.join(self.data_path, 'image', im_name))).astype(np.float32)/255
        image = torch.from_numpy(image_array.transpose(2,0,1))
        # mask
        img_cloth_high, mask, parse_mask = mask_parse(parse)
        parse_mask = parse_mask.astype('uint8')
        # parse7_occ
        parse7_s = ParseOneHot(parse, num_class=7)
        parse7_s = torch.from_numpy(parse7_s.astype(np.float32))
        parse7_occ = ParseOneHot(parse_mask, num_class=7).astype(np.float32)
        # hand
        hand_mask_arr = np.array(Image.open(osp.join(self.data_path, 'other', 'hand', im_name.replace("jpg", "png"))).convert("L")).astype(np.float32)
        hand_mask_arr = np.expand_dims(np.clip(hand_mask_arr,0,1), axis=2)
        # image_occ
        image_occ_arr = mask_image(mask, image.numpy().transpose(1,2,0))
        image_occ_arr = image_occ_arr * (1-hand_mask_arr) + image_array * hand_mask_arr
        image_occ = torch.from_numpy(image_occ_arr.transpose(2,0,1).astype(np.float32))
        # pose_map18
        pose_map18 = get_pose_map18(im_name, self.data_path, self.fine_height, self.fine_width, 5, self.transform)

        # ==================== mis_parse ============================
        parse_rm_cloth = parse
        parse_rm_cloth[np.where(parse==3)]=0
        parse_rm_cloth[np.where(parse==4)]=0
        parse_rm_cloth[np.where(parse==5)]=0
        mask = 1 - ((parse==1) + (parse==2) + (parse==6)) 
        pre_mloth_mask = warp_mloth_np * mask
        mis_parse = parse_rm_cloth + pre_mloth_mask * 3
        mis_parse[np.where(mis_parse>6)] = 3
        # mis_parse = np.zeros_like(mis_parse)
        mis_parse_tensor = torch.from_numpy(mis_parse.astype(np.float32)).unsqueeze(0)

        # ================= limb ==========================
        limb_mask = (parse==4) + (parse==5)
        # limb_mask = (parse_s==1) + (parse_s==4) + (parse_s==5)+ (parse_s==6)
        limb_mask = np.expand_dims(limb_mask, axis=2)
        limb_mask = np.concatenate((limb_mask, limb_mask, limb_mask), axis=2)
        image_limb = image_array * limb_mask
        
        image_limb = np.zeros_like(image_limb)

        limb = Image.fromarray(image_limb.astype('uint8'))
        image_limb = torch.from_numpy(image_limb.transpose(2,0,1)/255)
        # limb_patch
        scale = 8
        patch_height = 256 // scale
        patch_width = 192 // scale
        limb_patches = []
        for i in range(scale):
            for j in range(scale):
                limb_patch = np.array(limb.crop((j*patch_width, i*patch_height,(j+1)*patch_width, (i+1)*patch_height)))
                limb_patches.append(limb_patch)
        limb_patches = np.array(limb_patches).astype(np.float32)/255  # [64,32,24,3]
        limbs = limb_patches[0]
        for i in range(limb_patches.shape[0]):
            if i != 0:
                limbs = np.concatenate((limbs, limb_patches[i]), axis=2)
        limbs = limbs.transpose(2,0,1)    # [192,32,24]

        # tensor - [0,1]
        result = {
            'c_name':               c_name,                 # list
            'im_name':              im_name,                # list

            # input
            'cloth':                cloth_tenor,
            'mloth':                mloth_tensor,           # [b, 3, 256, 192]
            'pose_map18':           pose_map18,             # [b, 18, 256, 192]
            'parse7_occ':           parse7_occ,            # [b, 7, 256, 192]
            'image_occ':            image_occ,               # [b, 3, 256, 192]
            'mis_parse':            mis_parse_tensor,
            'limbs':               limbs,

            'image':                image,                  # [b, 3, 256, 192] 
            'parse':                parse,                  # [b, 1, 256, 192] 
            }          

        return result

    def __len__(self):
        return len(self.im_names)

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch