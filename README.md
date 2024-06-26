# Video-based Cloth-Changing Person Re-Identification

This repository contains implementation of training and testing baseline for **Video-based Cloth-Changing Person Re-ID (VCCRe-ID)**. This is part of the official implementation for the paper: [Temporal 3D Shape Modeling for Video-based Cloth-changing Person Re-Identification](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Nguyen_Temporal_3D_Shape_Modeling_for_Video-Based_Cloth-Changing_Person_Re-Identification_WACVW_2024_paper.html)

## News
- E-VCCR dataset will be released soon.

## 1. Features

#### Supported CNN backbones

- `c2dres50`: C2DResNet50
- `i3dres50`: I3DResNet50
- `ap3dres50`: AP3DResNet50
- `nlres50`: NLResNet50
- `ap3dnlres50`: AP3DNLResNet50

#### Summary of VCCRe-ID datasets

This baseline currently supports the public VCCRe-ID datasets: **VCCR**, **CCVID**, and **CCPG**.

| Dataset | Num.IDs | Num.Tracklets | Num.Clothes/ID | Public | Download |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Motion-ReID | 30 | 240 | - | X | - |
| CVID-reID | 90 | 2980 | - | X | - |
| SCCVRe-ID | 333 | 9620 | 2~37 | X | - |
| RCCVRe-ID | 34 | 6948 | 2~10 | X | - |
| CCPG | 200 | ~16k | - | Per Request | [project link](https://github.com/BNU-IVC/CCPG) |
| CCVID | 226 | 2856 | 2~5 | Yes | [link](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view?usp=sharing) |
| VCCR | 392 | 4384 | 2~10 | Yes | [link](https://drive.google.com/file/d/17qJPksE-Fk189KSHTPYQihMfnzXnHC6m/view) |

## 2. Running instructions

### Getting started

#### Create virtual environment

First, create a virtual environment for the repository
```bash
conda create -n vccreid python=3.8
```
then activate the environment 
```bash
conda activate vccreid
```


#### Clone the repository

```bash
git clone https://github.com/dustin-nguyen-qil/VCCReID-Baseline.git
```
Next, install the dependencies by running
...
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the datasets VCCR and CCVID following download links above
2. Create a folder named `data` inside the repository
3. Run the following command line (**Note**: replace the path to the folder storing the datasets and the dataset name)
```bash
python datasets/prepare.py --root "/media/dustin/DATA/Research/Video-based ReID" --dataset_name vccr
```

### Configuration options

Go to `./config.py` to modify configurations accordingly
- Dataset name
- Number of epochs
- Batch size
- Learning rate
- CNN backbone (according to model names above)
- Choice of loss functions

If training from checkpoint, copy checkpoint path and paste to RESUME in `./config.py`.

### Run baseline

Create a folder named `work_space` inside the repository, then create two subfolders named `save` and `output`.

```
data
work_space
|--- save
|--- output
main.sh
```

#### Run

```bash
bash main.sh
```

Trained model will be automatically saved to `work_space/save`.
Testing results will be automatically saved to `work_space/output`.

## Citation

If you find this repo helpful, please cite:

```bash
@InProceedings{Nguyen_2024_WACV,
    author    = {Nguyen, Vuong D. and Mantini, Pranav and Shah, Shishir K.},
    title     = {Temporal 3D Shape Modeling for Video-Based Cloth-Changing Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {173-182}
}
```


## Acknowledgement

Related repos: 
- [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID). 













