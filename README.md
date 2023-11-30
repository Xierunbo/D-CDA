# D-CDA
## SDNet
### Prerequisites
- Python 3.6
- PyTorch 1.8
- NVIDIA GPU + CUDA cuDNN
### Usage
- #### Train
````
python3 train.py --project PROJECT_NAME --noisy-train-dir NOISY_IMAGE_TRAIN_DIR --clean-train-dir CLEAN_IMAGE_TRAIN_DIR --noisy-valid-dir NOISY_IMAGE_VALID_DIR --clean-valid-dir CLEAN_IMAGE_VALID_DIR 
````
- #### test
````
python3 test.py --weights-dir SAVE_WEIGHT_DIR --clean-image-dir test_clean --noisy-image-dir test_noisy --save-dir test_img
````

## CAFNet
### Requirements

- Python 3.6

- Pytorch 1.4

- torchvision 0.5.0
### Train 

    python train.py

### Evaluate model performance

    python eval.py

### Visualization

    python visualization.py
