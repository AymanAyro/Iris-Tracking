# Iris Tracking Detection Model

![Eff3_00_00_27-00_00_43](https://github.com/user-attachments/assets/854e2de8-9439-411d-ab39-7ce93b1b72ab)
*Real-time iris tracking on webcam feed*

## Overview
This repository contains an end-to-end solution for iris detection and tracking using deep learning. The model leverages a modified EfficientNetB3 backbone with custom CNN layers to predict iris coordinates from RGB images. The entire pipeline includes data collection, augmentation, model training with TensorBoard monitoring, and real-time inference.

## Key Features
- Custom dataset collection using OpenCV (145 base images → 7,250 augmented samples)
- Advanced data augmentation pipeline with Albumentations
- Modified EfficientNetB3 architecture with custom head
- Mixed-precision training with AdamW optimizer
- Real-time webcam inference with tracking visualization
- Comprehensive TensorBoard monitoring

## Methodology

### 1. Data Collection & Annotation
- Captured 145 high-resolution eye images using OpenCV
- Annotated iris coordinates using LabelMe
- Generated segmentation masks and keypoint vectors

![Screenshot 2024-08-16 173349](https://github.com/user-attachments/assets/ad17411c-5e47-420f-a5b0-8e30f5b4936c)

### 2. Data Preprocessing
- Resized images to 224×224 (EfficientNet optimal resolution)
- Normalized pixels to [0,1] range
- Applied aggressive augmentation:
  
  ```python
  A.Compose([
      A.HorizontalFlip(p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
      A.RandomGamma(gamma_limit=(80, 120), p=0.5),
      A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
      A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
      A.CoarseDropout(max_holes=30, max_height=16, max_width=16, min_holes=10, p=0.5),
      A.CLAHE(p=0.3)
  ])
  ```

![image](https://github.com/user-attachments/assets/a0db1368-326e-4e57-a1a2-09f6f350d550)

### 3. Model Architecture

```python
base_model = EfficientNetB3(include_top=False, weights='imagenet')
base_model.layers[0] = Conv2D(40, 3, padding='same', activation='relu',
                              input_shape=(224, 224, 3), use_bias=False)

x = base_model.output
x = Conv2D(256, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Conv2D(4, (2, 2), activation='linear')(x)
output = GlobalAveragePooling2D()(x)

model = Model(base_model.input, output, name='Iris_Detection_ModelV4')
```

### 4. Training Configuration

- **Optimizer**: AdamW (lr=1e-3, β1=0.9, β2=0.937)
- **Loss**: Huber (δ=0.2)
- **Mixed Precision**: Enabled via LossScaleOptimizer
- **Metrics**: MSE, MAE
- **Train/Val/Test Split**: 80%/10%/10%
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)

### 5. Training Monitoring with TensorBoard

#### Key Metrics Tracked:
- **Epoch Loss**: Huber loss progression across training

![image](https://github.com/user-attachments/assets/fd92e3cc-99e1-4ae4-83d5-e14a8f4e1950)

- **Learning Rate**: Effective learning rate schedule

![image](https://github.com/user-attachments/assets/23c34c0e-40cb-4d46-9451-e009b7681e2b)

- **Beta Histogram**: Distribution of AdamW's β parameters

![image](https://github.com/user-attachments/assets/ddc050f5-14d2-4d2e-825c-97a0761beb61)

- **Bias Histogram**: Layer bias value distributions

![image](https://github.com/user-attachments/assets/09e951d6-ff5b-41c0-9cf3-c3f1ad554a02)

- **Epoch MSE**: Mean Squared Error progression

![image](https://github.com/user-attachments/assets/ec9480bc-cf12-4e58-9d09-0f0e25135353)

- **Epoch MAE**: Mean Absolute Error progression

![image](https://github.com/user-attachments/assets/5c4831ed-ab7a-4850-bc88-990c9984d06b)

### Challenges & Solutions

- **Limited Data**: Overcame with aggressive augmentation (50× multiplier)
- **Precision Requirements**: Addressed through Huber loss and batch normalization
- **Hardware Constraints**: Optimized using mixed precision training
- **Overfitting Prevention**: Implemented CoarseDropout and L2 regularization

### Future Work

- Expand dataset with diverse lighting conditions and eye colors
- Implement spatial-temporal modeling for smoother tracking
- Experiment with Vision Transformers (ViTs) for attention-based localization
- Develop mobile-optimized version using TFLite
- Integrate gaze estimation capabilities
- Implement 3D iris tracking using depth sensors

![10570198](https://github.com/user-attachments/assets/783cc695-c9fb-4eab-ac4a-447668550f2f)

---

## MIT License

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
