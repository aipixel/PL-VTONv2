import torch
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
from dataset import Dataset, Dataset2, DataLoader
from model import STNNet
from model import FlowModel_GRU as MCWNet
from model import ParseModel as HPENet
from model import Network as LTFNet
from visualization import save_images, Parse_7_to_1

torch.manual_seed(0)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Test3")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="C:/Users/Admin/Desktop/viton_code/data/viton")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--pair_setting", default="unpair", choices=['pair', 'unpair'])
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--shuffle", type=bool, default=True, help='shuffle input data')
    opt = parser.parse_args()
    return opt

def train_network(opt, train_loader, model_STN, model_MCW):
    model_STN.eval()
    model_MCW.eval()

    model_STN.cuda()
    model_MCW.cuda()

    save_dir = os.path.join('result', opt.name, opt.datamode, opt.pair_setting)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mloth_dir = os.path.join(save_dir, 'warp-mloth')
    if not os.path.exists(warp_mloth_dir):
        os.makedirs(warp_mloth_dir)

    num_data = len(os.listdir(os.path.join(opt.dataroot, opt.datamode, "cloth")))
    step = (num_data // opt.batch_size) + 1

    for step in tqdm(range(step)):
        inputs = train_loader.next_batch()
        c_name = inputs['c_name']                                  # list
        cloth = inputs['cloth'].cuda()                             # [b, 3, 256, 192] 
        mloth = inputs['mloth'].cuda()                             # [b, 1, 256, 192]
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_occ = inputs['parse7_occ'].cuda()                   # [b, 7, 256, 192]
        image_occ = inputs['image_occ'].cuda()                     # [b, 7, 256, 192]

        theta = model_STN(cloth, pose_map18, parse7_occ)

        grid_c = F.affine_grid(theta, cloth.size())
        grid_m = F.affine_grid(theta, mloth.size())
        pre_cloth = F.grid_sample(cloth, grid_c, padding_mode='border')
        pre_mloth = F.grid_sample(mloth, grid_m)

        flow, warp_cloth = model_MCW(pre_cloth, pose_map18, parse7_occ, image_occ)
        warp_mloth = F.grid_sample(pre_mloth, flow, mode='bilinear', padding_mode='border')

        save_images(warp_mloth, c_name, warp_mloth_dir)
        save_images(warp_cloth, c_name, warp_cloth_dir)

def train_network2(opt, train_loader, model_HPE, model_LTF):
    model_HPE.eval()
    model_LTF.eval()

    model_HPE.cuda()
    model_LTF.cuda()

    save_dir = os.path.join(opt.mid_data, opt.datamode, opt.pair_setting)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coarse_try_on_dir = os.path.join(save_dir, 'coarse-try-on')
    if not os.path.exists(coarse_try_on_dir):
        os.makedirs(coarse_try_on_dir)
    fine_try_on_dir = os.path.join(save_dir, 'fine-try-on')
    if not os.path.exists(fine_try_on_dir):
        os.makedirs(fine_try_on_dir)
    parse_t_dir = os.path.join(save_dir, 'parse_t')
    if not os.path.exists(parse_t_dir):
        os.makedirs(parse_t_dir)

    num_data = len(os.listdir(os.path.join(opt.dataroot, opt.datamode, "cloth")))
    step = (num_data // opt.batch_size) + 1

    for step in tqdm(range(step)):
        inputs = train_loader.next_batch()

        im_name = inputs['im_name']                                # list
        warp_cloth = inputs['cloth'].cuda()                        # [b, 3, 256, 192] 
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_occ = inputs['parse7_occ'].cuda()                   # [b, 7, 256, 192]
        image_occ = inputs['image_occ'].cuda()                     # [b, 7, 256, 192]
        mis_parse = inputs['mis_parse'].cuda()                     # [b, 7, 256, 192]
        limbs = inputs['limbs'].cuda()                             # [b, 7, 256, 192]
        
        parse7_t = model_HPE(warp_cloth, pose_map18, parse7_occ, image_occ, mis_parse)

        try_on_coarse, try_on_fine = model_LTF(limbs, warp_cloth, pose_map18, parse7_t, image_occ)

        save_images(Parse_7_to_1(parse7_t), im_name, parse_t_dir, type="parse")
        save_images(try_on_coarse, im_name, coarse_try_on_dir)
        save_images(try_on_fine, im_name, fine_try_on_dir)


if __name__ == "__main__":
    opt = get_opt()

    print("====================== 创建模型 ======================")
    model_STN = torch.nn.DataParallel(STNNet()).cuda()
    weight_STN = torch.load("./ckpt/STN.pth")
    model_STN.load_state_dict(weight_STN)

    model_MCW = torch.nn.DataParallel(MCWNet()).cuda()
    weight_MCW = torch.load("./ckpt/MCW.pth")
    model_MCW.load_state_dict(weight_MCW)

    model_HPE = torch.nn.DataParallel(HPENet()).cuda()
    weight_HPE = torch.load("./ckpt/HPE.pth")
    model_HPE.load_state_dict(weight_HPE)

    model_LTF = torch.nn.DataParallel(LTFNet()).cuda()
    weight_LTF = torch.load("./ckpt/LTF.pth")
    model_LTF.load_state_dict(weight_LTF)

    print("====================== 加载数据 ======================")
    # create dataset
    test_dataset = Dataset(opt)
    # create dataloader
    test_loader = DataLoader(opt, test_dataset)
    print("数据加载完成，测试集大小:", test_dataset.__len__())
    print("====================== 生成结果 ======================")
    with torch.no_grad():
        train_network(opt, test_loader, model_STN, model_MCW)

    # ================== 2 ===================
    opt.mid_data = os.path.join("result", opt.name)
    # create dataset
    test_dataset2 = Dataset2(opt)
    # create dataloader
    test_loader2 = DataLoader(opt, test_dataset2)
    with torch.no_grad():
        train_network2(opt, test_loader2, model_HPE, model_LTF)

