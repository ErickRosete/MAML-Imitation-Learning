# MAML-Imitation-Learning
Imitation learning using MAML in the metaworld benchmark

## Instructions

To run this program you need a working installation of Mujoco. https://www.roboti.us/index.html

This program uses the metaworld environment for learning and their provided policies,
more information of the installation in the following link: https://github.com/rlworkgroup/metaworld


## Program description

This program is based on ideas from the paper "One-Shot Visual Imitation Learning via Meta-Learning": https://arxiv.org/abs/1709.04905

To execute training, in this setup we meta-train an initial configuration such that after fine tuning with one demonstration of
a task it is able to perform appropiately, the purpose of this is to extend the knowledge from other application so that it can use this previous
experience to learn faster a new task.
```console
foo@bar:~$ python train.py --use-cuda --batch-size 32 --num-batches 500
```

To execute testing, in this setup we fine tune the meta-trained parameters to a specific task and test it on a new goal position in the same task.
```console
foo@bar:~$ python test.py --use-cuda --batch-size 32 
```


