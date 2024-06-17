# This code is the source code implementation for the paper "PDP-FD: Federated Knowledge Distillation Based on Personalized Differential Privacy ."



## Abstract
![输入图片说明](/imgs/2024-06-17/b8U9so7PKA9U8PB5.png)
Federated Learning (FL) is a distributed machine learning framework where each participant achieves model training by exchanging model parameters. However, when the server is dishonest, it may lead to the leakage of the private data of participants. Furthermore, FL is also affected by non-independent and identically distributed (Non-IID) data, resulting in a decrease in model accuracy. At the same time, Non-IID data may also lead to different privacy protection requirements among participants, making it difficult to handle them uniformly. In this paper, we propose a Federated Knowledge Distillation Based on Personalized Differential Privacy (PDP-FD) that considers both data heterogeneity and privacy protection problems. To solve the data heterogeneity problem, we adopt a network structure with a personalization layer and propose a strategy of dynamically adjusting the personalization layer. By adjusting the personalization layer, we can adequately preserve local features while better adapting to the data characteristics among different participants. To solve the problem of different privacy requirements among participants, we propose a personalized privacy budget allocation strategy, which adds appropriate noise based on the training state of the local model to achieve personalized privacy protection. Finally, the experimental results show the effectiveness of our mechanism and its superior performance over other differential privacy FL schemes.


# Experimental Environment

```
- Pytorch==3.9
- scikit-learn==1.1.3
- numpy==1.23.4
— opacus==1.3.0
- matplotlib==3.7.1
- easydict==1.10
- dill==0.3.6
- pandas==1.5.3

```
## Experimental Setup

### Datasets and Models:

- **Datasets**: Experiments were carried out using CIFAR-10 and MNIST datasets. CIFAR-10 consists of 60,000 color images in 10 classes, with 6,000 images per class. The MNIST dataset includes 60,000 training and 10,000 test images of handwritten digits, each of size 28x28 pixels. For auxiliary datasets, CIFAR-100 and USPS datasets were used for CIFAR-10 and MNIST, respectively.

- **Models**: Various ResNet models (ResNet-18, ResNet-34, ResNet-50) and a CNN model were utilized. The ResNet models differ in the number of layers, and CNN was likely a simpler, custom structure appropriate for the task.

### Implementation Details:

- **Local Training**: Each client performed local training with its dataset. The local epoch count was set to 2, with a batch size of 128, and the learning rate was set at 0.01.

- **Communication Rounds**: There were 100 communication rounds involving 10 client participants.

- **Distillation Specifics**: The distillation process, which involves transferring knowledge from the ensemble model to the global model, also ran for 2 epochs with a batch size of 128.

- **Personalization Layer**: The approach included a personalization layer, dynamically adjusted between 1 to 4 layers in depth during the experiments.

### Experimental Procedure:

- **Model Training**: Each user trained their local model using their dataset, following which the model parameters were sent to a central server.

- **Parameter Aggregation**: At the server, parameters from different clients were aggregated to update the global model.

- **Differential Privacy (DP) Application**: Noise was added during the model update process to ensure differential privacy.

- **Knowledge Distillation**: The global model was then refined through knowledge distillation techniques to enhance its performance and generalization capabilities.

### Privacy Cost Comparison:

- The paper also compared the privacy costs of different privacy budget allocation strategies, demonstrating that the proposed method could achieve comparable or better accuracy at lower privacy costs compared to other methods.

## Python Files
### 1. `Fed.py`

This file contains functions related to federated learning, specifically for averaging the model weights from different clients. Key functions include:

- **FedAvg**: Averages the weights of models from all clients when each client has the same number of samples. This is a simple averaging function where weights are summed and then divided by the number of clients.

- **customFedAvg**: Similar to `FedAvg` but allows for custom weight assignments to different models, although in the snippet it's implemented as simple averaging.

### 2. `models_readme.md`

It describes several functions found in `Fed.py` such as `FedAvg`, a refined version of this function for cases where clients have different numbers of samples, and a function incorporating differential privacy into the averaging process.

### 3. `Nets.py`

This file contains neural network models defined using PyTorch. The snippet suggests it includes implementations of various architectures, those used in experiments related to the documents or for training within a federated learning framework.

### 4. `rdp_accountant.py`

This file deals with the accounting of differential privacy using the Rényi Differential Privacy (RDP) framework. It includes functions for calculating the RDP of an algorithm and for determining how much privacy budget has been spent. This is critical in scenarios where differential privacy is applied, ensuring that the privacy guarantee limits are adhered to during computation.

### 5. `test.py`

This script is designed for testing models, within the context of federated learning. It includes functionality to evaluate a model's performance on a dataset, calculating metrics like accuracy and loss. The function `test_img` suggests it be used specifically for image datasets, considering parameters like global models, test datasets, and user-defined arguments for execution.

### 6. `Update.py`

This file contains code that likely handles updates to datasets or models within a learning framework. The `DatasetSplit` class mentioned in the snippet is used to handle subsets of data corresponding to specific clients in a federated learning setting. This is useful for simulating a more realistic environment where each client only has access to a portion of the overall dataset.

##  Experimental Results
The experimental results demonstrate that the PDP-FD method significantly outperforms other methods in terms of model accuracy, convergence speed, and privacy cost across the tested datasets (CIFAR-10 and MNIST). This success is attributed to the dynamic adjustment of the personalization layer, which better adapts to local data heterogeneities, and a personalized privacy budget allocation strategy that effectively balances privacy protection with model utility. These improvements allow the PDP-FD method to achieve higher model performance with lower privacy costs compared to existing methods, thereby confirming the efficacy of the proposed approach in managing the challenges posed by non-IID data and varying privacy requirements in federated learning environments.
![输入图片说明](/imgs/2024-06-17/SzKBuDwPnpid4IjX.png)
![输入图片说明](/imgs/2024-06-17/ULK73PMoes7ScJIQ.png)![输入图片说明](/imgs/2024-06-17/RsjvJc9Qsr3az9vQ.png)


```
## Update log

```
- {24.06.17} Uploaded overall framework code and readme file
```
