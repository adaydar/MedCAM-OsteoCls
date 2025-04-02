#MedCAM-OsteoCls

This repo contains the Official Pytorch implementation of our paper:

MEDCAM-OSTEOCLS: MEDical Context Aware Multimodal Classification of Knee Osteoarthritis (Accepted at ICASSP-25)

![MedCAM_OsteoCls_architecture](./MedCAM_OsteoCls_architecture.png) Figure: Overall schematic of the proposed MedCAM-OsteoCls model with (a) VGG-19-TE +Fully Connected (FC) Network, (b) the CG-SSP module and (c) the XMRCA module.

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    1. Kindly check "https://github.com/adaydar/MtRA-Unet/tree/main" repository for dataset preparation.
    2. Then kindly check the "config" file before running the training and testing code.

Training and Testing:

Prepare the dataset and then run the following command for training:

    python3 train.py

For Testing, run

    python3 test.py
