# LEVERAGING LABEL HIERARCHIES FOR FEW-SHOT EVERYDAY SOUND RECOGNITION
## Introduction

This repo contains supplement materials for the manuscript "LEVERAGING LABEL HIERARCHIES FOR FEW-SHOT EVERYDAY SOUND RECOGNITION" for Detection and Classification of Acoustic Scenes and Events (DCASE) workshop 2022. The repo impliments several off-the-shelf few-shot algorithms including linear regression, prototypical networks, matching networks on ESC-50 in a simple, usr-friendly manner.

## Preparation

Before implimenting our experiments, please 
1. Download [ESC-50](https://github.com/karolpiczak/ESC-50) in a custom path and modify the path in ``cfg/DATASOURCE/esc50.yaml``
2. Create the environment using
``pip install -r requirement.txt``

## Run
0. This repo is empowered by [hydra](https://github.com/facebookresearch/hydra) and [wandb](https://github.com/wandb/client), please read through their doc in the first beginning.
1. Modify the experimental setting via.
``cfg/fewshot.yaml``
2. Now we are ready to go! Feel free to use our code by
``python fewshot.py``


## Cite
If you utilise our code, please cite our work