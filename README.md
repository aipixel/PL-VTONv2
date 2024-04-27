## Limb-Aware Virtual Try-On Network With Progressive Clothing Warping, IEEE Transactions on Multimedia'23.
Official code for IEEE Transactions on Multimedia 2023 paper 'Limb-Aware Virtual Try-On Network With Progressive Clothing Warping'

we propose a novel virtual try-on network named PL-VTON, where three sub-modules are designed to generate high-quality try-on results, including Progressive Clothing Warping, Person Parsing Estimator, and Limb-aware Texture Fusion. On the one hand, PL-VTON explicitly models the location and size of the in-shop clothing and utilizes a two-stage alignment strategy to estimate the fine-grained clothing warping progressively. On the other hand, PL-VTON adopts limb-aware guidance to generate realistic limb details during the texture fusion between the warped clothing and the human body.

[[Paper]](https://ieeexplore.ieee.org/abstract/document/10152500)

[[Checkpoints]](https://drive.google.com/file/d/18KvqkWWbjI_GHkqF5HZes0RNB233DHPG/view?usp=share_link)

## Notice
Our method is an extension of our previous work on the conference version: (https://github.com/xyhanHIT/PL-VTON). Improvements made in this article include: 1) In Progressive Clothing Warping, we adopt a new pre-alignment network to regress dynamic parameters of translation and scaling, which improves the robustness of parameter acquisition and avoids the negative impact of special human poses on the transformation parameters. 2) Considering the fit of the person wearing clothing in real scenarios, we propose a novel gravity-aware loss to better handle the clothing edges and make the warped result more realistic. 3) We introduce a non-limb target parsing map that contains the semantic information of the warped clothing in the middle process of Person Parsing Estimator, which serves as a geometric prior and assists in the smooth transition of predicting the target parsing map. 4) We further optimize the clothing-agnostic person representation by the semantic correction according to the person’s parsing result to better retain the person’s other characteristics except for clothing and limbs in the try-on result. 5) We conduct more qualitative and quantitative experiments. Specifically, we additionally introduce Structural Similarity (SSIM) and Peak Signal to Noise Ratio (PSNR) to further measure the quality of the generated try-on result. Then, we add three new ablation studies to evaluate each proposed contribution of our PL-VTON in more detail. Finally, additional try-on results of our method on the MPV, VITON-HD, and Dress Code datasets are further provided.

## Pipeline
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/pipeline.png)

## Environment
python 3.7

torch 1.9.0+cu111

torchvision 0.10.0+cu111

## Dataset
For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

## Inference
1. Download the checkpoints from [here](https://drive.google.com/file/d/1kaVWi2zeeeJv5-xs9Ea8oWjzkzKdGeCH/view?usp=sharing).

2. Get [VITON dataset](https://github.com/xthan/VITON).

3. Run the "test.py".
```bash
python test.py
```
**Note that** the results of our pretrained model are guaranteed in VITON dataset only.

## Sample Try-on Results
  
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/experiment1.png)
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/experiment2.png)
![image](https://github.com/xyhanHIT/PL-VTONv2/blob/main/images/experiment3.png)

## License
The use of this code is restricted to non-commercial research and educational purposes.

## Citation
If you use our code or models, please cite with:
```
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
@inproceedings{han2018viton,
  title={VITON: An Image-Based Virtual Try-On Network},
  author={Han, Xintong and Wu, Zuxuan and Wu, Zhe and Yu, Ruichi and Davis, Larry S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7543--7552},
  year={2018}
}
```