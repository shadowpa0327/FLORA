# FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer

Code for the paper: [WACV 2024] [FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer](https://arxiv.org/abs/2311.03912)




## Data Preparation (ImageNet)
We use the Imagenet-1k for our main benchmark of our compression. Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

If you already have it, you can just simply skip this step.

## Docker

We run all of our experiment on using the pytorch docker images `nvcr.io/nvidia/pytorch:22.02-py3`. In this repo, we provide a script for user to build the docker image and run the container easily. 

### Build docker image
```
cd docker
./build_docker.sh
```
### Run the docker
Before we start running the docker, we have to first set up the path of the ImageNet-1k dataset in `docker/common.sh`. By doing so, the script will automatically mount the dataset while you running the container. More specifically, 
 modify the variable `IMAGENET1k_PATH` in the `docker/common.sh` to the path of you ImageNet-1k dataset. After setting, run the following command to activate the docker container.
```
./run.sh
```
After running this command, a docker container will be activated and the whole FLORA repository will be mounted. 

## Install dependencies

```
pip install -r requirements.txt
```


## Prepare Pretrained Weights
Before we start, we have to first get the pretrained weights. The following list are the url of checkpoints that copy from the [official deit](https://github.com/facebookresearch/deit/blob/main/README_deit.md) and [official swin](https://github.com/microsoft/Swin-Transformer).

Juse simply use the `wget` or `curl` to get the weights. The following is the example to get the deit-s pretrained weights using `wget`
```
wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
```

Then, run the following command to preprocess the pretrained weight for following supernet training
```python
python supernet.py --cfg configs/lr_deit/supernet/lr_deit_base_supernet.yaml \
--pretrained ./deit_base_patch16_224-b5f2ef4d.pth
```

## Prepare Distillation Logits
In our searching framework, we conduct the knowledge distillation in an offline manner following [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT), user can refer to TinyViT for more detail. To generate the logits using the uncompressed model itself as a teacher, run the following command:
```python
python -m torch.distributed.launch --nproc_per_node 8 save_logits.py \
--cfg configs/teacher/deit_b.yaml \
--data-path /imagenet \
--batch-size 128 \
--eval \
--resume ./deit_base_patch16_224-b5f2ef4d.pth \
--opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits_deit_b
```
The above we run the inference to generate the prediction logits given the training data with data augmentation and save it.

## Search Space Filtering
To be updated

## Train Supernet
To train the supernet, run the following command:
```python
python -m torch.distributed.launch --nproc_per_node 8 main.py \
--cfg configs/lr_deit/supernet/lr_deit_base_supernet.yaml \
--data-path /imagenet \
--batch-size 128 \
--resume lr_deit_base_supernet.pth \
--opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits_deit_b
```

## Evolutionary Search
```python
python -m torch.distributed.launch --nproc_per_node=1  evolution_search.py \
--cfg [Path to supernet configs] \
--pretrained [Path to the weight of trained supernet] \
--param-limits [Upper bound of the target GFLOPs] \
--min-param-limits [Lower bound of the target GFLOPs]\
--data-path /imagenet
```

## Get the subnet
Once you get the config of seearched subnet, you can leverage the `subnet.py` to slice the weights of subnet from the weight of supernet by running the following command:
```python
python subnet.py --cfg [Path to subnet configs] \
--pretrained [Path to the weights of supernet]
```

After you get the weights of subnet then you can use the `main.py` to evaluate the accuracy of the searched subnet.
```python
python -m torch.distributed.launch --nproc_per_node=1  \
main.py --cfg [Path to the subnet configs] \
--pretrained [Path to the weights of subnet] \
--eval \
--data-path /imagenet
```

## Checkpoint of subnets

Table provide below present the evaluation result of our search low-rank architecture and its checkpoint.

| name | acc@1 | acc@5 | #params | checkpoint url |
| --- | --- | --- | --- | --- |
| lr_deit_small | 72.2 | 94.62 | 12.6M | [google drive](https://drive.google.com/file/d/1j5ZJ0xO0looiClAwxOMm_AxdZIQUjzTH/view?usp=drive_link) |
| lr_deit_base | 81.8 | 95.54 | 38M| [google drive](https://drive.google.com/file/d/104tGK8hyINF3IeTilAbftkshT-EzRM2k/view?usp=drive_link) |



## Citation
```bibtex
@InProceedings{Chang_2024_WACV,
    author    = {Chang, Chi-Chih and Sung, Yuan-Yao and Yu, Shixing and Huang, Ning-Chi and Marculescu, Diana and Wu, Kai-Chiang},
    title     = {FLORA: Fine-Grained Low-Rank Architecture Search for Vision Transformer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2482-2491}
}
```

## Acknowledgement
Our code is building on top of [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT).
