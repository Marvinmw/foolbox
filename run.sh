#!/bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4-23:00:00
#SBATCH -p bigmem
#SBATCH --qos=qos-bigmem
#SBATCH -C skylake
#SBATCH -J  mediate_images_cw
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu


module purge
module load math/Keras/2.1.6-foss-2018a-TensorFlow-1.8.0-Python-3.6.4
module load lib/TensorFlow/1.8.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
module load vis/matplotlib/2.1.2-foss-2018a-Python-3.6.4
pip install --upgrade --user  keras sklearn tqdm

srun -n 1 --pty python ./generateAdvImgs.py -m lenet -d mnist -a FGSM CW > ./deepxplore_f.info
srun -n 1 --pty python ./generateAdvImgs.py -m deepxplore -d fashion_mnist -a FGSM CW > ./deepxplore_f.info
srun -n 1 --pty python ./generateAdvImgs.py -m vgg -d cifar10 -a FGSM CW > ./vgg.info

srun -n 1 --pty python ./generateAdvImgs.py -m netinnet -d cifar10 -a FGSM CW > ./netinnet.info
srun -n 1 --pty python ./generateAdvImgs.py -m mlp -d mnist -a FGSM CW > ./mlp_m.info
srun -n 1 --pty python ./generateAdvImgs.py -m mlp -d fashion_mnist -a FGSM CW > ./mlp_f.info

srun -n 1 --pty python ./generateAdvImgs.py -m lenet -d fashion_mnist -a FGSM CW > ./lenet_f.info

srun -n 1 --pty python ./generateAdvImgs.py -m deepxplore -d mnist -a FGSM CW > ./deepxplore_m.info