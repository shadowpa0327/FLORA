# FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer

Code for the paper: [WACV 2024] [FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer]()




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
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
```


## Prepare Distillation Logits
In our searching framework, we conduct the knowledge distillation in an offline manner following [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT), user can refer to TinyViT for more detail. To generate the logits using the uncompressed model itself as a teacher, run the following command:
```
python -m torch.distributed.launch --nproc_per_node 8 save_logits.py --cfg configs/teacher/deit_s.yaml --data-path /imagenet --batch-size 128 --eval --resume ./deit_small_patch16_224-cd65a155.pth --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits_deit_s
```
The above we run the inference to generate the prediction logits given the training data with data augmentation and save it.

## Train Supernet
To train the supernet, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg configs/lr_deit/supernet/lr_deit_base_supernet_v2_local_search.yaml --data-path /dev/shm/imagenet --batch-size 128 --resume weights/lr_deit_base_supernet.pth --output deit_base_local_search --tag t35_v50_s25 --use-wandb --opts DISTILL.TEACHER_LOGITS_PATH /dev/shm/teacher_logits/
```

## Citation
```
To be done
```

## Acknowledgement
Our code is building on top of [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT).