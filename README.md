## Limb-Aware Virtual Try-On Network With Progressive Clothing Warping, IEEE Transactions on Multimedia'23.
Official code for TMM 2023 paper 'Limb-Aware Virtual Try-On Network With Progressive Clothing Warping'   

![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/results1.png)   
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/results2.png)

We propose a novel virtual try-on network named PL-VTON, where three sub-modules are designed to generate high-quality try-on results, including Progressive Clothing Warping, Person Parsing Estimator, and Limb-aware Texture Fusion. On the one hand, PL-VTON explicitly models the location and size of the in-shop clothing and utilizes a two-stage alignment strategy to estimate the fine-grained clothing warping progressively. On the other hand, PL-VTON adopts limb-aware guidance to generate realistic limb details during the texture fusion between the warped clothing and the human body.

[[Paper]](https://arxiv.org/abs/2503.14074)

[[Checkpoints]](https://drive.google.com/file/d/18KvqkWWbjI_GHkqF5HZes0RNB233DHPG/view?usp=share_link)

## Notice
Our method is an extension of our previous work on the conference version: https://github.com/xyhanHIT/PL-VTON, and improvements made in this article include: 
* a new pre-alignment network to regress dynamic parameters of translation and scaling
* a novel gravity-aware loss 
* a non-limb target parsing map
* the optimization of the clothing-agnostic person representation
* more qualitative and quantitative experiment results

## Pipeline
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/pipeline.png)

## Environment
python 3.8

torch 2.0.1+cu118

torchvision 0.15.2+cu118

## Dataset
For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

You also need to download the data about hands from [here](https://drive.google.com/file/d/1VbzXS6vYumRoUaVp0PRXvB_1d54aqxM6/view?usp=drive_link).

## Inference
1. Download the checkpoints from [here](https://drive.google.com/file/d/1y98JcPR1TQ-qQCD7rwV8k11BRqBE4Jr-/view?usp=drive_link).

2. Get [VITON dataset](https://github.com/xthan/VITON).

3. Run the "test.py".
```bash
python test.py
```

## License
The use of this code is restricted to non-commercial research and educational purposes.

## Our Team's Researches
- **[[ACM MM'22] PL-VTON](https://github.com/xyhanHIT/PL-VTON)** - Progressive Limb-Aware Virtual Try-On
- **[[IEEE TMM'23] PL-VTONv2](https://github.com/aipixel/PL-VTONv2)** - Limb-Aware Virtual Try-On Network With Progressive Clothing Warping
- **[[ACM MM'24] SCW-VTON](https://github.com/xyhanHIT/SCW-VTON)** - Shape-Guided Clothing Warping for Virtual Try-On

## Citation
If you use our code or models, please cite with:
```bibtex
@article{zhang2023limb,
  title={Limb-aware virtual try-on network with progressive clothing warping},
  author={Zhang, Shengping and Han, Xiaoyu and Zhang, Weigang and Lan, Xiangyuan and Yao, Hongxun and Huang, Qingming},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
@inproceedings{han2022progressive,
  title={Progressive Limb-Aware Virtual Try-On},
  author={Han, Xiaoyu and Zhang, Shengping and Liu, Qinglin and Li, Zonglin and Wang, Chenyang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2420--2429},
  year={2022}
}
```
