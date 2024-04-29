#coding=utf-8
import torch
from PIL import Image
from PIL import ImageDraw
import os.path as osp
import numpy as np
import json

def ParseOneHot(x, num_class=None):
    h, w = x.shape
    x = x.reshape(-1)
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((x.shape[0], num_class))
    ohx[range(x.shape[0]), x] = 1
    ohx = ohx.reshape(h,w, ohx.shape[1])
    return ohx.transpose(2,0,1).astype(np.float32)

def ParseOneHotReverse(x):
    h, w = x.shape[1], x.shape[2]
    x = x.reshape(7,-1).transpose(1,0)
    res = [np.argmax(item, axis=0) for item in x]
    res = np.array(res).reshape(h, w)
    return res.astype('uint8')

def mask_parse(parse):
    hand = np.where((parse==4) | (parse==5))
    cloth = np.where(parse==3)
    bias = 3
    vertical_high = max(cloth[0])
    vertical_high = np.clip((vertical_high + bias),0,256)
    vertical_low = min(cloth[0])
    vertical_low = np.clip((vertical_low - bias),0,256)
    level_high1 = max(cloth[1])
    level_high1 = np.clip((level_high1 + bias),0,192)
    level_low1 = min(cloth[1])
    level_low1 = np.clip((level_low1 - bias),0,192)
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
    # --------- other regions -----------
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

    # -------------
    # 0：bg
    # 1：hair
    # 2：face
    # 3：cloth / mask
    # 4：left arm
    # 5：right arm
    # pants / shoes
    # -------------
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