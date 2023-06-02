# ERA_Session5

# Task : MNIST Digit Recognition using Fully Convolutional Neural Network

Introduction
This project aims to train and test a Fully Convolutional Neural Network (FCN) for the task of digit recognition on the MNIST dataset. The MNIST dataset consists of handwritten digit images from 0 to 9, total number of images 60,000.

# Requirements

Python 

TensorFlow 

NumPy

Matplotlib (for visualization)

# Installation
1. Clone the repository

   !git clone https://github.com/sunandhini96/ERA_Session5.git
   
2. Installing the required packages

   pip install -r requirements.txt
   
 # Usage
 
 --> model.py : defined the architecture of the model
 
 --> utils.py :  defining the training and testing functions
 
 --> S5.ipynb :  data downloading and applying transformations and training and testing the model code and output of the model
                 
  python model.py
  
  python utils.py
  
  # Results:
  
  Trained model over 20 epochs 
  
  **Summary of the model**:
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model=Net().to(device)
  summary(model, input_size=(1, 28, 28))
  
  *Output* :
  
Train: Loss=1.4625 Batch_id=117 Accuracy=30.55: 100%|██████████| 118/118 [00:25<00:00,  4.57it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 1.2455, Accuracy: 6505/10000 (65.05%)

Epoch 2
Train: Loss=0.1609 Batch_id=117 Accuracy=87.63: 100%|██████████| 118/118 [00:27<00:00,  4.32it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.1237, Accuracy: 9643/10000 (96.43%)

Epoch 3
Train: Loss=0.0895 Batch_id=117 Accuracy=95.38: 100%|██████████| 118/118 [00:26<00:00,  4.44it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.1038, Accuracy: 9659/10000 (96.59%)

Epoch 4
Train: Loss=0.0382 Batch_id=117 Accuracy=96.52: 100%|██████████| 118/118 [00:26<00:00,  4.47it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0576, Accuracy: 9820/10000 (98.20%)

Epoch 5
Train: Loss=0.0436 Batch_id=117 Accuracy=97.33: 100%|██████████| 118/118 [00:26<00:00,  4.40it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0446, Accuracy: 9861/10000 (98.61%)

Epoch 6
Train: Loss=0.0791 Batch_id=117 Accuracy=97.61: 100%|██████████| 118/118 [00:26<00:00,  4.47it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0400, Accuracy: 9870/10000 (98.70%)

Epoch 7
Train: Loss=0.0644 Batch_id=117 Accuracy=97.86: 100%|██████████| 118/118 [00:26<00:00,  4.49it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0400, Accuracy: 9871/10000 (98.71%)

Epoch 8
Train: Loss=0.0294 Batch_id=117 Accuracy=98.10: 100%|██████████| 118/118 [00:26<00:00,  4.46it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0306, Accuracy: 9904/10000 (99.04%)

Epoch 9
Train: Loss=0.0141 Batch_id=117 Accuracy=98.21: 100%|██████████| 118/118 [00:26<00:00,  4.48it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0314, Accuracy: 9900/10000 (99.00%)

Epoch 10
Train: Loss=0.0865 Batch_id=117 Accuracy=98.45: 100%|██████████| 118/118 [00:26<00:00,  4.41it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0291, Accuracy: 9911/10000 (99.11%)

Epoch 11
Train: Loss=0.0171 Batch_id=117 Accuracy=98.43: 100%|██████████| 118/118 [00:26<00:00,  4.44it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0326, Accuracy: 9895/10000 (98.95%)

Epoch 12
Train: Loss=0.0120 Batch_id=117 Accuracy=98.55: 100%|██████████| 118/118 [00:26<00:00,  4.52it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0405, Accuracy: 9879/10000 (98.79%)

Epoch 13
Train: Loss=0.0158 Batch_id=117 Accuracy=98.62: 100%|██████████| 118/118 [00:26<00:00,  4.43it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0280, Accuracy: 9902/10000 (99.02%)

Epoch 14
Train: Loss=0.0092 Batch_id=117 Accuracy=98.67: 100%|██████████| 118/118 [00:26<00:00,  4.46it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0257, Accuracy: 9912/10000 (99.12%)

Epoch 15
Train: Loss=0.0077 Batch_id=117 Accuracy=98.75: 100%|██████████| 118/118 [00:26<00:00,  4.46it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0230, Accuracy: 9923/10000 (99.23%)

Epoch 16
Train: Loss=0.0064 Batch_id=117 Accuracy=99.02: 100%|██████████| 118/118 [00:25<00:00,  4.61it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0206, Accuracy: 9925/10000 (99.25%)

Epoch 17
Train: Loss=0.0245 Batch_id=117 Accuracy=99.00: 100%|██████████| 118/118 [00:25<00:00,  4.57it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0198, Accuracy: 9934/10000 (99.34%)

Epoch 18
Train: Loss=0.0329 Batch_id=117 Accuracy=99.06: 100%|██████████| 118/118 [00:25<00:00,  4.65it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0200, Accuracy: 9934/10000 (99.34%)

Epoch 19
Train: Loss=0.0281 Batch_id=117 Accuracy=99.04: 100%|██████████| 118/118 [00:25<00:00,  4.71it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0200, Accuracy: 9930/10000 (99.30%)

Epoch 20
Train: Loss=0.0342 Batch_id=117 Accuracy=99.11: 100%|██████████| 118/118 [00:24<00:00,  4.78it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0199, Accuracy: 9931/10000 (99.31%)
  
  
  **Plotting the curves**:
  
  ![image](https://github.com/sunandhini96/ERA_Session5/assets/63030539/7d847ab8-3428-4e32-b735-c1e27c3b0366)

  
  Observations :
  
  Training accuracy: 99.11 %
  
  Testing accuracy: 99.31 %
  
  
  
  

