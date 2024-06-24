# Celebrity Face Recognition

This repository contains the code developed for celebrity face recognition using a subset of the Celebrity Face Recognition Dataset. The following steps were undertaken to achieve accurate recognition:

## Steps

1. **Face Cropping with MTCNN**:
   - Faces were cropped from images using MTCNN (Multi-Task Cascaded Convolutional Neural Network). To execute face cropping, run `cropper_MCNN.py`.

2. **Model Fine-Tuning**:
   - Several models were fine-tuned using the cropped dataset to optimize performance for face recognition.

3. **Sample Reweighing**:
   - Due to imbalance and noise in the dataset, sample reweighing techniques were applied, as described in [this paper](https://arxiv.org/pdf/1803.09050).

4. **Ensembling Predictions**:
   - Predictions from multiple fine-tuned models were ensembled using `combined_predict.py` to generate the final result.

## Results

The approach resulted in an overall accuracy of 86.18%, which was the highest achieved for the competition.

### Usage

To replicate the process:
- Execute `cropper_MCNN.py` to crop faces using MTCNN.
- Fine-tune models on the cropped dataset and train using 'train_using_<base_model>.py'.
- Use `combined_predict.py` to ensemble predictions for final accuracy assessment.


