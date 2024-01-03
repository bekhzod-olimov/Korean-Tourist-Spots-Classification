# Korean-Tourist-Spots-Classification
This repository contains training, inference processess of deep learning-based image classification model and analysis of the trained model using [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) on the [Korean Tourist Spot Dataset](https://github.com/DGU-AI-LAB/Korean-Tourist-Spot-Dataset).

* Create conda environment from yml file using the following script:

a) Create a virtual environment using yml file:

```python
conda env create -f environment.yml
```

Then activate the environment using the following command:
```python
conda activate speed
```

b) Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n speed python=3.9
```

- Activate the environment using the following command:

```python
conda activate speed
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

* Download dataset from the [link](https://github.com/DGU-AI-LAB/Korean-Tourist-Spot-Dataset).

* Train the model using the following arguments:

![image](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/022d11fa-d189-4cdf-ad7d-2eab6e87e118)

```python

python main.py --dataset_name "korean_landmarks" --batch_size 32 --epochs 30

```
* Inference process with trained model using the following arguments:

![image](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/89ddb8b8-58e8-48d1-a908-91bf11c48554)

```python

python inference.py --dataset_name "korean_landmarks" --save_path "saved_models" --dls_dir "saved_dls"

```

* Inference Results:
  
        Predictions:
  ![korean_landmarks_preds](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/7cd585e2-6f65-4afe-ba01-87a41e9a51d4)
  ![korean_landmarks_preds_2](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/a5aefc7a-6a4c-49aa-b779-37a9ca525a82)
  ![korean_landmarks_preds_3](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/1f533082-be87-486a-91a2-0cb05c963488)

        GradCam:
  ![korean_landmarks_gradcam](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/84ef0632-eefb-409a-af33-2f50853339e4)
  ![korean_landmarks_gradcam_2](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/61eec758-38fb-458e-b9b6-06fb2078f75f)
  ![korean_landmarks_gradcam_3](https://github.com/bekhzod-olimov/Korean-Tourist-Spots-Classification/assets/50166164/3a8ce17d-048f-45fb-a170-ab295d6a7d47)



