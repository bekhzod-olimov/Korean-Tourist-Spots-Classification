# Korean-Tourist-Spots-Classification
This repository contains training, inference processess of deep learning-based image classification model and analysis of the trained model using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) on the [Korean Tourist Spot Dataset](https://github.com/DGU-AI-LAB/Korean-Tourist-Spot-Dataset).

* Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

* Download dataset from the [link](https://github.com/DGU-AI-LAB/Korean-Tourist-Spot-Dataset).

* Train the model using the following arguments:

![image](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/022d11fa-d189-4cdf-ad7d-2eab6e87e118)

```python

python main.py --dataset_name "korean_landmarks" --batch_size 32 --epochs 30

```
