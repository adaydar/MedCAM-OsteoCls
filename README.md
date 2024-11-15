#MedCAM-OsteoCls

This repo contains the Official Pytorch implementation of our paper:

MEDCAM-OSTEOCLS: MEDICAL CONTEXT AWARE MULTIMODAL CLASSIFICATION OF KNEE OSTEOARTHRITIS (UNDER REVIEW at ICASSP-25)

![MedCAM_OsteoCls_architecture](./MedCAM_OsteoCls_architecture.png)/

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    1. Kindly check "https://github.com/adaydar/MtRA-Unet/tree/main" repository for dataset preparation.
    2. Then kindly check the "config" file for code preparation.

Training and Testing:

Prepare the dataset and then run the following command for pretrain:

    python3 train.py

For Testing, run

    python3 test.py
