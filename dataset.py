#coding=utf-8
from ast import excepthandler
from importlib.abc import PathEntryFinder
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image
from PIL import ImageDraw
from visualization import save_images
import os.path as osp
import numpy as np
import json
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from skimage import transform
def ParseOneHot(x, num_class=None):
    h, w = x.shape
    x = x.reshape(-1)
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((x.shape[0], num_class))
    ohx[range(x.shape[0]), x] = 1
    ohx = ohx.reshape(h,w, ohx.shape[1])
    return ohx.transpose(2,0,1)

def ParseOneHotReverse(x):
    h, w = x.shape[1], x.shape[2]
    x = x.reshape(7,-1).transpose(1,0)
    res = [np.argmax(item, axis=0) for item in x]
    res = np.array(res).reshape(h, w)
    return res.astype('uint8')

def mask_parse(parse):
    hand = np.where((parse==4) | (parse==5))
    cloth = np.where(parse==3)
    vertical_high = max(cloth[0])
    vertical_low = min(cloth[0])
    level_high1 = max(cloth[1])
    level_low1 = min(cloth[1])
    try:
        level_high2 = max(hand[1])
        level_low2 = min(hand[1])
        if level_high1 < level_high2:
            level_high = level_high2
        else:
            level_high = level_high1
        if level_low1 < level_low2:
            level_low = level_low1
        else:
            level_low = level_low2
    except:
        level_high = level_high1
        level_low = level_low1
    mask = np.zeros((vertical_high-vertical_low, level_high-level_low))
    mask_level1 = np.ones((vertical_high-vertical_low, level_low))
    mask_level2 = np.ones((vertical_high-vertical_low, parse.shape[1]-level_high))
    mask = np.concatenate((mask_level1, mask, mask_level2), axis=1)
    mask_vertical1 = np.ones((vertical_low, parse.shape[1]))
    mask_vertical2 = np.ones((parse.shape[0]-vertical_high, parse.shape[1]))
    mask = np.concatenate((mask_vertical1, mask, mask_vertical2), axis=0)
    if parse.ndim == 3:
        mask = np.repeat(mask, 3, 2)

    other_mask = (parse==1) + (parse==2) + (parse==6)
    mask = 1 - (1-mask) * (1-other_mask)
    
    res = parse * mask
    color = (1-mask) * 3
    res = res + color
    return vertical_high-vertical_low, mask, res

def mask_image(mask, image):
    mask = np.expand_dims(mask, 2)
    mask = np.repeat(mask, 3, 2)
    res = image * mask
    return res

def ParseFine(parse):
    parse_background = (parse==0)
    parse_hair = (parse==2)
    parse_cloth1 = (parse==5)
    parse_cloth2 = (parse==6)
    parse_cloth3 = (parse==7)
    parse_low_cloth1 = (parse==8)
    parse_low_cloth2 = (parse==9)
    parse_cloth4 = (parse==10)
    parse_cloth5 = (parse==11)
    parse_low_cloth3 = (parse==12)
    parse_face = (parse==13)
    parse_left_hand = (parse==14)
    parse_right_hand = (parse==15)
    parse_leg1 = (parse==16)    
    parse_leg2 = (parse==17)   
    parse_shoe1 = (parse==18)    
    parse_shoe2 = (parse==19) 

    parse = parse_background *0 + \
        parse_hair * 1 + \
        parse_face * 2 + \
        (parse_cloth1 + parse_cloth2 + parse_cloth3 + parse_cloth4 + parse_cloth5) * 3 + \
        parse_left_hand * 4 + \
        parse_right_hand * 5 + \
        (parse_low_cloth1 + parse_low_cloth2 + parse_low_cloth3 + parse_leg1 + parse_leg2 + parse_shoe1 + parse_shoe2) * 6 
    
    return parse.astype("uint8")

# pose_map18
def get_pose_map18(im_name, data_path, fine_height, fine_width, radius, transform):
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    with open(osp.join(data_path, 'pose', pose_name), 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1,3))
    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, fine_height, fine_width)
    r = radius
    im_pose = Image.new('L', (fine_width, fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (fine_width, fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]
    pose_map18 = pose_map.numpy()*0.5 + 0.5
    return pose_map18

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
        # cloth名字
        c_name = self.c_names[index]
        # image名字
        im_name = self.im_names[index]
        # cloth
        cloth = np.array(Image.open(osp.join(self.data_path, 'cloth', c_name))).astype('uint8')
        cloth_tenor = torch.from_numpy(cloth.astype(np.float32).transpose(2,0,1)/255)
        # mloth
        mloth = np.array(Image.open(osp.join(self.data_path, 'cloth-mask', c_name))).astype('uint8')
        mloth_tensor = torch.from_numpy(mloth.astype(np.float32)/255).unsqueeze(0)
        # parse
        parse = np.array(Image.open(osp.join(self.data_path, 'image-parse', im_name.replace('.jpg', '.png')))).astype('uint8')
        parse = ParseFine(parse) # [0-19] -> [0-6]
        # image
        image_array = np.array(Image.open(osp.join(self.data_path, 'image', im_name))).astype(np.float32)
        image = torch.from_numpy(image_array.transpose(2,0,1)/255)
        
        # 获得parse的服装高度、遮挡掩膜和遮挡的parse
        img_cloth_high, mask, parse_mask = mask_parse(parse)
        parse_mask = parse_mask.astype('uint8')
        
        # 将parse和遮挡的parse进行独热编码
        parse7_s = ParseOneHot(parse, num_class=7)
        parse7_s = torch.from_numpy(parse7_s.astype(np.float32))
        parse7_occ = ParseOneHot(parse_mask, num_class=7).astype(np.float32)

        # 获得遮挡的image
        image_occ = mask_image(mask, image.numpy().transpose(1,2,0))
        image_occ = torch.from_numpy(image_occ.transpose(2,0,1).astype(np.float32))

        # pose_map18
        pose_map18 = get_pose_map18(im_name, self.data_path, self.fine_height, self.fine_width, 5, self.transform)

        # image_mloth
        image_mloth = torch.from_numpy((parse==3).astype(np.float32)).unsqueeze(0)

        # image_cloth
        parse_cloth = (parse == 3).astype(np.float32)
        pcm = torch.from_numpy(parse_cloth).unsqueeze(0)
        image_cloth = image * pcm + (1 - pcm) 

        # limb
        limb_mask = (parse==4) + (parse==5)
        # limb_mask = (parse_s==1) + (parse_s==4) + (parse_s==5)+ (parse_s==6)
        limb_mask = np.expand_dims(limb_mask, axis=2)
        limb_mask = np.concatenate((limb_mask, limb_mask, limb_mask), axis=2)
        image_limb = image_array * limb_mask
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

        # _tensor的范围均为[0,1]
        result = {
             # 显示
            'c_name':               c_name,                 # list
            'im_name':              im_name,                # list
            # 输入
            'cloth':                cloth_tenor,
            'mloth':                mloth_tensor,           # [b, 3, 256, 192]
            'pose_map18':           pose_map18,             # [b, 18, 256, 192]
            'parse7_occ':           parse7_occ,            # [b, 7, 256, 192]
            'image_occ':            image_occ,               # [b, 3, 256, 192]
            'limbs:':               limbs,
            # 标签
            'image_cloth':          image_cloth,            # [b, 3, 256, 192]
            'image_mloth':          image_mloth,            # [b, 1, 256, 192]
            # 参考
            'image':                image,                  # [b, 3, 256, 192] 
            'parse':                parse,                  # [b, 1, 256, 192] 
            }          

        return result

    def __len__(self):
        return len(self.im_names)

class Dataset2(data.Dataset):
    def __init__(self, opt):
        super(Dataset2, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.mid_data = opt.mid_data
        self.datamode = opt.datamode # train or test or self-defined
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
        # cloth名字
        c_name = self.c_names[index]
        # image名字
        im_name = self.im_names[index]
        # cloth
        cloth = np.array(Image.open(osp.join(self.mid_data, self.datamode, self.pair_setting, 'warp-cloth', c_name))).astype('uint8')
        cloth_tenor = torch.from_numpy(cloth.astype(np.float32).transpose(2,0,1)/255)
        # mloth
        warp_mloth_np = np.array(Image.open(osp.join(self.mid_data, self.datamode, self.pair_setting, 'warp-mloth', c_name))).astype('uint8')
        mloth_tensor = torch.from_numpy(warp_mloth_np.astype(np.float32)/255).unsqueeze(0)
        # parse
        parse = np.array(Image.open(osp.join(self.data_path, 'image-parse', im_name.replace('.jpg', '.png')))).astype('uint8')
        parse = ParseFine(parse) # [0-19] -> [0-6]
        # image
        image_array = np.array(Image.open(osp.join(self.data_path, 'image', im_name))).astype(np.float32)
        image = torch.from_numpy(image_array.transpose(2,0,1)/255)
        
        # 获得parse的服装高度、遮挡掩膜和遮挡的parse
        img_cloth_high, mask, parse_mask = mask_parse(parse)
        parse_mask = parse_mask.astype('uint8')

        # 将parse和遮挡的parse进行独热编码
        parse7_s = ParseOneHot(parse, num_class=7)
        parse7_s = torch.from_numpy(parse7_s.astype(np.float32))
        parse7_occ = ParseOneHot(parse_mask, num_class=7).astype(np.float32)

        # 获得遮挡的image
        image_occ = mask_image(mask, image.numpy().transpose(1,2,0))
        image_occ = torch.from_numpy(image_occ.transpose(2,0,1).astype(np.float32))

        # pose_map18
        pose_map18 = get_pose_map18(im_name, self.data_path, self.fine_height, self.fine_width, 5, self.transform)

        # image_mloth
        image_mloth = torch.from_numpy((parse==3).astype(np.float32)).unsqueeze(0)

        # image_cloth
        parse_cloth = (parse == 3).astype(np.float32)
        pcm = torch.from_numpy(parse_cloth).unsqueeze(0)
        image_cloth = image * pcm + (1 - pcm) 

        # ==================== 错位数据 ============================
        parse_rm_cloth = parse
        parse_rm_cloth[np.where(parse==3)]=0
        parse_rm_cloth[np.where(parse==4)]=0
        parse_rm_cloth[np.where(parse==5)]=0
        mask = 1 - ((parse==1) + (parse==2) + (parse==6)) 
        pre_mloth_mask = warp_mloth_np/255 * mask
        mis_parse = parse_rm_cloth + pre_mloth_mask * 3
        mis_parse[np.where(mis_parse>6)] = 3
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

        # _tensor的范围均为[0,1]
        result = {
             # 显示
            'c_name':               c_name,                 # list
            'im_name':              im_name,                # list
            # 输入
            'cloth':                cloth_tenor,
            'mloth':                mloth_tensor,           # [b, 3, 256, 192]
            'pose_map18':           pose_map18,             # [b, 18, 256, 192]
            'parse7_occ':           parse7_occ,            # [b, 7, 256, 192]
            'image_occ':            image_occ,               # [b, 3, 256, 192]
            'mis_parse':            mis_parse_tensor,
            'limbs':               limbs,
            # 标签
            'image_cloth':          image_cloth,            # [b, 3, 256, 192]
            'image_mloth':          image_mloth,            # [b, 1, 256, 192]
            # 参考
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


if __name__ == "__main__":
    print("Check the dataset...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "../data/")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True, help='shuffle input data')
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"

    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)
   
    for step, inputs in enumerate(data_loader.data_loader):
        # 显示
        c_name = inputs['c_name']                                  # list
        im_name = inputs['im_name']                                # list
        image = inputs['image'].cuda()                             # [b, 3, 256, 192]
        # 输入
        cloth = inputs['cloth'].cuda()                             # [b, 3, 256, 192]
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_occ = inputs['parse7_occ'].cuda()                   # [b, 7, 256, 192]
        # image_occ = inputs['image_occ'].cuda()                     # [b, 3, 256, 192]
        # 计算Loss
        # pre_mloth = inputs['pre_mloth'].cuda()           # [b, 1, 256, 192]
        image_cloth = inputs['image_cloth'].cuda()                 # [b, 3, 256, 192]
        # image_mloth = inputs['image_mloth'].cuda()       # [b, 1, 256, 192]                         # [b, 3, 256, 192]



        # ========== 输入测试 =========
        pre_cloth = cloth.cpu().numpy()
        pose_map18 = pose_map18.cpu().numpy()
        parse7_occ = parse7_occ.cpu().numpy()
        # image_occ = image_occ.cpu().numpy()
        print("pre_cloth:", np.min(pre_cloth), np.max(pre_cloth), pre_cloth.shape, pre_cloth.dtype)
        print("pose_map18:", np.min(pose_map18), np.max(pose_map18), pose_map18.shape, pose_map18.dtype)
        print("parse7_occ:", np.min(parse7_occ), np.max(parse7_occ), parse7_occ.shape, parse7_occ.dtype)
        # print("image_occ:", np.min(image_occ), np.max(image_occ), image_occ.shape, image_occ.dtype)

        pre_cloth = pre_cloth[0].transpose(1,2,0)
        # image_occ = image_occ[0].transpose(1,2,0)
        parse7_occ = parse7_occ[0]
        parse_mask = ParseOneHotReverse(parse7_occ)

        pose_map = pose_map18[0][0]
        for i in range(1, len(pose_map18[0])):
            pose_map = pose_map + pose_map18[0][i]

        print("parse_mask:", np.min(parse_mask), np.max(parse_mask), parse_mask.shape, parse_mask.dtype)
        print("pose_map:", np.min(pose_map), np.max(pose_map), pose_map.shape, pose_map.dtype)
        plt.subplot(1,4,1)
        plt.imshow(pre_cloth)
        plt.subplot(1,4,2)
        plt.imshow(parse_mask)
        plt.subplot(1,4,3)
        plt.imshow(pose_map)
        plt.show()

        # ========= parse ============
        # parse7_s = parse7_s.cpu().numpy()[0]
        # print("parse7_s:", np.min(parse7_s), np.max(parse7_s), parse7_s.shape, parse7_s.dtype)
        # parse_s = ParseOneHotReverse(parse7_s)
        # print("im_name:", im_name[0])
        # plt.imshow(parse_s)
        # plt.show()
        # exit()
        # # ========== 标签测试 =========
        # pre_mloth = pre_mloth.cpu().numpy()
        # image_cloth = image_cloth.cpu().numpy()
        # image_mloth = image_mloth.cpu().numpy()
        # print("pre_mloth:", np.min(pre_mloth), np.max(pre_mloth), pre_mloth.shape, pre_mloth.dtype)
        # print("image_cloth:", np.min(image_cloth), np.max(image_cloth), image_cloth.shape, image_cloth.dtype)
        # print("image_mloth:", np.min(image_mloth), np.max(image_mloth), image_mloth.shape, image_mloth.dtype)

        # plt.subplot(1,4,1)
        # plt.imshow(pre_mloth[0][0])
        # plt.subplot(1,4,2)
        # plt.imshow(image_cloth[0].transpose(1,2,0))
        # plt.subplot(1,4,3)
        # plt.imshow(image_mloth[0][0])
        # plt.show()
        # exit()


        exit()

