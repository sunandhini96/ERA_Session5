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
  
  Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.2281 Batch_id=117 Accuracy=52.08: 100%|██████████| 118/118 [00:26<00:00,  4.39it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.2424, Accuracy: 9266/10000 (92.66%)

Epoch 2
Train: Loss=0.1809 Batch_id=117 Accuracy=93.81: 100%|██████████| 118/118 [00:27<00:00,  4.32it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0849, Accuracy: 9741/10000 (97.41%)

Epoch 3
Train: Loss=0.1854 Batch_id=117 Accuracy=96.35: 100%|██████████| 118/118 [00:27<00:00,  4.32it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0676, Accuracy: 9791/10000 (97.91%)

Epoch 4
Train: Loss=0.1071 Batch_id=117 Accuracy=96.90: 100%|██████████| 118/118 [00:27<00:00,  4.33it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0466, Accuracy: 9850/10000 (98.50%)

Epoch 5
Train: Loss=0.1879 Batch_id=117 Accuracy=97.60: 100%|██████████| 118/118 [00:29<00:00,  4.04it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0472, Accuracy: 9843/10000 (98.43%)

Epoch 6
Train: Loss=0.0993 Batch_id=117 Accuracy=97.69: 100%|██████████| 118/118 [00:27<00:00,  4.32it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0440, Accuracy: 9860/10000 (98.60%)

Epoch 7
Train: Loss=0.0933 Batch_id=117 Accuracy=98.06: 100%|██████████| 118/118 [00:27<00:00,  4.23it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0403, Accuracy: 9878/10000 (98.78%)

Epoch 8
Train: Loss=0.1320 Batch_id=117 Accuracy=98.23: 100%|██████████| 118/118 [00:27<00:00,  4.36it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0317, Accuracy: 9899/10000 (98.99%)

Epoch 9
Train: Loss=0.0296 Batch_id=117 Accuracy=98.35: 100%|██████████| 118/118 [00:27<00:00,  4.25it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0324, Accuracy: 9886/10000 (98.86%)

Epoch 10
Train: Loss=0.0209 Batch_id=117 Accuracy=98.50: 100%|██████████| 118/118 [00:27<00:00,  4.23it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0340, Accuracy: 9901/10000 (99.01%)

Epoch 11
Train: Loss=0.0091 Batch_id=117 Accuracy=98.60: 100%|██████████| 118/118 [00:27<00:00,  4.24it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0337, Accuracy: 9876/10000 (98.76%)

Epoch 12
Train: Loss=0.0238 Batch_id=117 Accuracy=98.70: 100%|██████████| 118/118 [00:27<00:00,  4.25it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0294, Accuracy: 9897/10000 (98.97%)

Epoch 13
Train: Loss=0.0102 Batch_id=117 Accuracy=98.75: 100%|██████████| 118/118 [00:27<00:00,  4.30it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0350, Accuracy: 9888/10000 (98.88%)

Epoch 14
Train: Loss=0.1072 Batch_id=117 Accuracy=98.81: 100%|██████████| 118/118 [00:28<00:00,  4.20it/s]Adjusting learning rate of group 0 to 1.0000e-02.

Test set: Average loss: 0.0268, Accuracy: 9913/10000 (99.13%)

Epoch 15
Train: Loss=0.0357 Batch_id=117 Accuracy=98.83: 100%|██████████| 118/118 [00:27<00:00,  4.30it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0271, Accuracy: 9909/10000 (99.09%)

Epoch 16
Train: Loss=0.0042 Batch_id=117 Accuracy=99.13: 100%|██████████| 118/118 [00:27<00:00,  4.29it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0231, Accuracy: 9920/10000 (99.20%)

Epoch 17
Train: Loss=0.0773 Batch_id=117 Accuracy=99.19: 100%|██████████| 118/118 [00:28<00:00,  4.16it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0228, Accuracy: 9922/10000 (99.22%)

Epoch 18
Train: Loss=0.0372 Batch_id=117 Accuracy=99.17: 100%|██████████| 118/118 [00:27<00:00,  4.28it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0230, Accuracy: 9923/10000 (99.23%)

Epoch 19
Train: Loss=0.0504 Batch_id=117 Accuracy=99.20: 100%|██████████| 118/118 [00:27<00:00,  4.24it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0221, Accuracy: 9926/10000 (99.26%)

Epoch 20
Train: Loss=0.0542 Batch_id=117 Accuracy=99.18: 100%|██████████| 118/118 [00:27<00:00,  4.35it/s]Adjusting learning rate of group 0 to 1.0000e-03.

Test set: Average loss: 0.0223, Accuracy: 9922/10000 (99.22%)


  
  
  **Plotting the curves**:
  
![image](https://github.com/sunandhini96/ERA_Session5/assets/63030539/eb01f728-66e4-40ce-98c7-c88c04e6cc7a)


  
  Observations :
  
  Training accuracy: 99.18 %
  
  Testing accuracy: 99.22 %
  
  
  
  

