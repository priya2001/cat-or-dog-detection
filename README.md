# Cats vs Dogs Classification 🐱🐶

## Project Overview

This project builds a **Deep Learning CNN (Convolutional Neural Network)** model to classify images as either **cats** or **dogs**. The model learns visual features from thousands of images and can accurately predict the animal type from new, unseen images.

**Real-world Application:** Automated image classification, content management systems, social media filtering, pet identification apps, and computer vision systems.

---

## Objective

Build a predictive model that:
- Correctly classifies images as cat or dog with high accuracy
- Learns visual patterns (edges, textures, shapes, features)
- Generalizes well to unseen images
- Handles overfitting through regularization techniques
- Provides confidence scores for predictions

---

## Dataset

**Source:** [Kaggle - Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

**Dataset Structure:**
```
PetImages/
├── Cat/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ... (thousands of cat images)
└── Dog/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ... (thousands of dog images)
```

### Dataset Characteristics:
- **Total Images:** Thousands of high-quality pet images
- **Classes:** 2 (Cat = 0, Dog = 1)
- **Image Size:** Variable (resized to 256×256 during preprocessing)
- **Channels:** 3 (RGB color images)
- **Image Quality:** Mix of professional and user-uploaded photos
- **Data Imbalance:** Relatively balanced between cats and dogs

---

## Project Workflow

```
1. DATA LOADING
   └─ Download dataset from Kaggle using kagglehub
   
2. DATA EXPLORATION
   ├─ List dataset structure
   ├─ Check image directories
   └─ Verify labels for both classes
   
3. DATA CLEANING
   ├─ Identify corrupted images (not readable as images)
   ├─ Check for non-image files (Thumbs.db, etc.)
   ├─ Remove problematic files
   └─ Ensure data quality before training
   
4. DATA PREPROCESSING
   ├─ Load images from directories
   ├─ Infer labels from folder names (Cat/Dog)
   ├─ Resize to 256×256 pixels
   ├─ Create batches (batch_size=32)
   ├─ Normalize pixel values (0-1 range)
   └─ Split: 80% training, 20% validation
   
5. FEATURE NORMALIZATION
   ├─ Pixel values: 0-255 → 0-1
   │  └─ Divide by 255.0
   ├─ Standardization
   └─ Ensures gradient stability during training
   
6. MODEL ARCHITECTURE DESIGN
   ├─ Conv2D Layer 1: 32 filters, 3×3 kernel
   ├─ BatchNormalization
   ├─ MaxPooling2D
   ├─ Conv2D Layer 2: 64 filters, 3×3 kernel
   ├─ BatchNormalization
   ├─ MaxPooling2D
   ├─ Conv2D Layer 3: 128 filters, 3×3 kernel
   ├─ BatchNormalization
   ├─ MaxPooling2D
   ├─ Flatten
   ├─ Dense(128) + Dropout(0.1)
   ├─ Dense(64) + Dropout(0.1)
   └─ Dense(1) + Sigmoid [Output]
   
7. MODEL COMPILATION
   ├─ Optimizer: Adam (adaptive learning rate)
   ├─ Loss: Binary Crossentropy (2 classes)
   └─ Metrics: Accuracy
   
8. MODEL TRAINING
   ├─ Epochs: 10
   ├─ Batch size: 32 (implicit)
   ├─ Validation split: 20%
   ├─ Track: Loss and Accuracy
   └─ Monitor overfitting
   
9. EVALUATION
   ├─ Generate loss graphs (train vs validation)
   ├─ Generate accuracy graphs (train vs validation)
   ├─ Analyze overfitting
   └─ Check convergence
   
10. PREDICTION
    ├─ Load test image
    ├─ Resize to 256×256
    ├─ Normalize pixel values
    ├─ Reshape to batch format (1, 256, 256, 3)
    └─ Get prediction probability
    
11. DECISION
    └─ If probability > 0.5 → Dog (1)
       If probability ≤ 0.5 → Cat (0)
```

---

## Technologies & Libraries

### **Core Libraries:**
- **Python 3.x** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **OpenCV (cv2)** - Image processing
- **Kaggle API** - Dataset access

### **Specific Tools Used:**

| Tool | Purpose | Usage |
|------|---------|-------|
| `kagglehub` | Dataset download | Access Kaggle datasets |
| `keras.utils.image_dataset_from_directory()` | Load image data | Batch loading with labels |
| `Conv2D` | Convolution operation | Extract spatial features |
| `MaxPooling2D` | Spatial reduction | Downsampling, feature selection |
| `BatchNormalization` | Normalize activations | Stabilize training, prevent overfitting |
| `Dropout` | Regularization | Reduce overfitting (10% dropout) |
| `Flatten` | Reshape for dense layers | Convert 3D to 1D |
| `Dense` | Fully connected layer | Classification layers |
| `ReLU activation` | Non-linearity | Learn complex patterns |
| `Sigmoid activation` | Output probability | Binary classification [0, 1] |
| `Adam optimizer` | Gradient descent | Adaptive learning rate |
| `Binary Crossentropy` | Loss function | Binary classification loss |
| `cv2.imread()` | Read image | Load test images |
| `matplotlib.pyplot` | Visualization | Plot loss and accuracy |

---

## CNN Model Architecture

### **Architecture Overview:**
```
INPUT: Image (256×256×3 RGB)
    ↓
BLOCK 1:
├─ Conv2D(32 filters, 3×3) + ReLU
├─ BatchNormalization
└─ MaxPooling(2×2)
    ↓
BLOCK 2:
├─ Conv2D(64 filters, 3×3) + ReLU
├─ BatchNormalization
└─ MaxPooling(2×2)
    ↓
BLOCK 3:
├─ Conv2D(128 filters, 3×3) + ReLU
├─ BatchNormalization
└─ MaxPooling(2×2)
    ↓
FLATTEN: Convert to 1D vector
    ↓
CLASSIFICATION:
├─ Dense(128) + ReLU + Dropout(0.1)
├─ Dense(64) + ReLU + Dropout(0.1)
└─ Dense(1) + Sigmoid
    ↓
OUTPUT: Probability [0, 1]
(0 = Cat, 1 = Dog)
```

### **Layer-by-Layer Details:**

#### **Convolutional Layers:**
```python
Conv2D(filters, kernel_size=(3,3), activation='relu')
```
- **Filter 1:** 32 filters → Learns edges, basic shapes
- **Filter 2:** 64 filters → Learns textures, patterns
- **Filter 3:** 128 filters → Learns complex features (eyes, ears, fur)

- **Kernel Size:** 3×3 → Scans small neighborhood
- **Activation:** ReLU → Introduces non-linearity

#### **Batch Normalization:**
```python
BatchNormalization()
```
- Normalizes layer outputs to mean=0, std=1
- Accelerates training
- Reduces internal covariate shift
- Acts as regularization (slight overfitting reduction)

#### **MaxPooling:**
```python
MaxPooling2D(pool_size=(2,2), strides=2)
```
- Reduces spatial dimensions by 50%
- Takes maximum value from each 2×2 window
- Retains important features
- Reduces computation
- Provides translation invariance

#### **Flatten:**
```python
Flatten()
```
- Converts 3D feature maps to 1D vector
- Input to fully connected layers
- Example: (32, 32, 128) → (131,072,)

#### **Dense Layers:**
```python
Dense(128, activation='relu')
Dense(64, activation='relu')
Dense(1, activation='sigmoid')
```
- First dense: 128 neurons (learns high-level features)
- Second dense: 64 neurons (refinement)
- Output: 1 neuron + Sigmoid (probability)

#### **Dropout:**
```python
Dropout(0.1)
```
- Randomly drops 10% of neurons during training
- Prevents co-adaptation
- Reduces overfitting
- Improves generalization

---

## Data Preprocessing

### **1. Image Loading and Directory Organization**
```python
keras.utils.image_dataset_from_directory(
    directory='PetImages/',
    labels='inferred',         # Cat/Dog from folder names
    label_mode='int',          # 0 or 1
    batch_size=32,             # Process 32 at a time
    image_size=(256, 256),     # Resize all images
    validation_split=0.2,      # 20% for validation
    subset='training'          # This gets training subset
)
```

### **2. Pixel Value Normalization**
```python
def process(image, label):
    image = tf.cast(image / 255., tf.float32)  # 0-255 → 0-1
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

**Why Normalize?**
- Neural networks converge faster with values in [0, 1]
- Prevents large pixel values from dominating
- Improves gradient flow
- Increases numerical stability

### **3. Data Augmentation Considerations**
Current model doesn't use augmentation, but it could improve results:
```python
# Could add in future iterations:
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])
```

### **4. Train-Validation Split**
```
Total Data
├── 80% → Training Set (8000 images)
└── 20% → Validation Set (2000 images)
```
- Validation set monitors overfitting during training
- Kept separate to evaluate generalization
- Same 80-20 split for both datasets

---

## Training Process

### **Training Configuration:**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds
)
```

### **Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Optimizer** | Adam | Adaptive learning rates |
| **Learning Rate** | Auto (Adam) | Gradient descent speed |
| **Loss Function** | Binary Crossentropy | Binary classification |
| **Batch Size** | 32 | Training sample batches |
| **Epochs** | 10 | Full dataset passes |
| **Activation (Hidden)** | ReLU | Non-linearity |
| **Activation (Output)** | Sigmoid | Probability [0, 1] |

### **What Happens Each Epoch:**
```
1. Forward Pass: Images → predictions
2. Calculate Loss: Compare predictions vs actual
3. Backward Pass: Compute gradients
4. Update Weights: Adjust using Adam optimizer
5. Validation: Check performance on validation set
6. Repeat: Next epoch with updated weights
```

### **What to Monitor:**
```
Good Training:
├─ Training loss: Decreases
├─ Validation loss: Decreases or plateaus
├─ Training accuracy: Increases toward 100%
└─ Gap between train & val: Small

Overfitting (Why BatchNorm + Dropout help):
├─ Training loss: Decreases
├─ Validation loss: Increases ← BAD
├─ Training accuracy: High
└─ Validation accuracy: Lower ← Overfitting

Regularization Techniques Used:
├─ BatchNormalization: Normalizes layer inputs
├─ Dropout: Random neuron deactivation
└─ Sufficient data: More images = better generalization
```

---

## Model Performance

### **Expected Performance:**
```
Training Accuracy:      85-95%
Validation Accuracy:    80-90%
Training Loss:          Low (< 0.3)
Validation Loss:        Low (< 0.4)
```

### **Performance Metrics:**
- **Accuracy:** % of correct predictions
- **Loss:** Binary Crossentropy value
  - 0 = Perfect predictions
  - Higher = Worse

### **Evaluation Graphs:**

**Accuracy Graph:**
```
Accuracy (%)
│
100│ ╱╱╱ Training (red)
   │╱╱╱ Validation (blue)
75 │╱ Should follow similar trend
   │
50 │
   └─────────────────
     0   5   10 Epochs

Good: Both curves increase and converge
Warning: Large gap = Overfitting
```

**Loss Graph:**
```
Loss
│
 3│ Training (red)
  │ Validation (blue)
 2│ ╲╲╲
  │  ╲╲╲ Should decrease
 1│   ╲╲ and stabilize
  │    ╲
 0│_____╲___________
   0   5   10 Epochs

Good: Both curves decrease
Warning: Val loss increasing = Overfitting
```

---

## How to Use the Model

### **Making Predictions on New Images:**

```python
# Step 1: Load image
test_img = cv2.imread('path/to/image.jpg')

# Step 2: Resize to 256×256
test_img = cv2.resize(test_img, (256, 256))

# Step 3: Reshape for model (add batch dimension)
test_input = test_img.reshape((1, 256, 256, 3))

# Step 4: Normalize
test_input = test_input / 255.0

# Step 5: Predict
prediction = model.predict(test_input)

# Step 6: Interpret
if prediction[0] > 0.5:
    print("🐶 DOG (Probability: {:.2%})".format(prediction[0]))
else:
    print("🐱 CAT (Probability: {:.2%})".format(1 - prediction[0]))
```

### **Prediction Process:**
```
Raw Image (variable size)
    ↓ [Resize to 256×256]
Resized Image
    ↓ [Normalize 0-1]
Normalized Input
    ↓ [Add batch dimension: (1,256,256,3)]
Model Input
    ↓ [Pass through 3 Conv blocks]
Feature Maps (encoded features)
    ↓ [Flatten & Dense layers]
Probability Score (0-1)
    ↓ [If > 0.5 → Dog, else → Cat]
Final Prediction
```

---

## Overfitting & Regularization

### **What is Overfitting?**
```
Model memorizes training data instead of learning generalizable patterns

Signs:
├─ Training accuracy: Very high (95%+)
├─ Validation accuracy: Much lower (75%)
└─ Large gap indicates overfitting
```

### **Techniques Used to Reduce Overfitting:**

1. **Batch Normalization**
   - Normalizes layer outputs
   - Reduces internal covariate shift
   - Acts as mild regularization

2. **Dropout (0.1)**
   - Randomly deactivates 10% of neurons
   - Prevents co-adaptation
   - Forces network to learn redundant features

3. **Sufficient Data**
   - Thousands of diverse images
   - More data = better generalization

### **Other Techniques (Could be Added):**
```python
# Data Augmentation
data_aug = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# L1/L2 Regularization
Dense(128, kernel_regularizer=l2(0.001))

# Early Stopping
EarlyStopping(monitor='val_loss', patience=3)

# Learning Rate Reduction
ReduceLROnPlateau(monitor='val_loss', factor=0.5)
```

---

## Project Files

```
Cats_v_Dogs_Classification.ipynb
├── Cell 1: Kaggle Colab badge link
├── Cell 2: Import kagglehub
├── Cell 3: Download dataset
├── Cell 4: Print dataset path
├── Cell 5: Import TensorFlow libraries
├── Cell 6-7: Create train & validation datasets
├── Cell 8: Normalize pixel values
├── Cell 9: Define CNN model architecture
├── Cell 10: Print model summary
├── Cell 11: Compile model
├── Cell 12: Train model (10 epochs)
├── Cell 13: Data cleaning (corrupted images)
├── Cell 14-15: Plot accuracy graphs
├── Cell 16-17: Plot loss graphs
├── Cell 18: Overfitting explanation
├── Cell 19-20: Duplicate accuracy plots
├── Cell 21-22: Duplicate loss plots
├── Cell 23: Import cv2 for image reading
├── Cell 24: Load test image
├── Cell 25: Display test image
├── Cell 26: Check image shape
├── Cell 27: Resize image
├── Cell 28: Reshape for model
└── Cell 29: Make prediction
```

---

## Installation & Setup

### **Requirements:**
```
Python 3.7+
GPU recommended (for faster training)
```

### **Install Dependencies:**
```bash
pip install tensorflow keras numpy matplotlib opencv-python kagglehub
```

### **For Google Colab (Recommended):**
```
- All libraries pre-installed
- Free GPU access
- Easy Kaggle integration
- No installation needed
```

---

## How to Run

### **Step 1: Set Up Environment**
```python
# Ensure Kaggle API is configured
# In Colab: kagglehub handles it automatically
```

### **Step 2: Download Dataset**
```python
import kagglehub
path = kagglehub.dataset_download("karakaggle/kaggle-cat-vs-dog-dataset")
```

### **Step 3: Run Data Preparation**
- Create train and validation datasets
- Normalize pixel values
- Verify data is loaded

### **Step 4: Build Model**
- Define sequential architecture
- Add layers (Conv2D, BatchNorm, MaxPooling, Dense, Dropout)
- Compile with Adam optimizer

### **Step 5: Train Model**
- Run training for 10 epochs
- Monitor loss and accuracy
- Validate on validation set

### **Step 6: Evaluate**
- Generate loss graphs
- Generate accuracy graphs
- Analyze overfitting

### **Step 7: Make Predictions**
- Load test image
- Resize and normalize
- Get prediction probability
- Interpret result

---

## Expected Results

### **Model Performance:**
```
Training Results (10 epochs):
├─ Final Training Accuracy: 85-95%
├─ Final Validation Accuracy: 80-90%
├─ Training Loss: 0.2-0.4
└─ Validation Loss: 0.3-0.5

Sample Predictions:
├─ Cat image → [0.15] → "🐱 CAT"
├─ Dog image → [0.85] → "🐶 DOG"
└─ Confidence scores increase with training
```

### **What to Look For in Graphs:**
```
Accuracy:
- Both training & validation increase
- Convergence around epoch 8-10
- Small gap between train & validation

Loss:
- Both decrease over epochs
- Stabilize in later epochs
- Validation close to training
```

---

## Customization & Improvements

### **Model Enhancements:**

1. **More Convolutional Blocks:**
   ```python
   # Add 4th or 5th conv block for deeper learning
   model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
   ```

2. **Transfer Learning:**
   ```python
   from tensorflow.keras.applications import VGG16
   base_model = VGG16(weights='imagenet')
   # Much faster & better results
   ```

3. **Data Augmentation:**
   ```python
   data_augmentation = Sequential([
       RandomFlip("horizontal"),
       RandomRotation(0.2),
       RandomZoom(0.2),
       RandomContrast(0.2),
   ])
   ```

4. **Hyperparameter Tuning:**
   ```python
   epochs = 20  # More training
   batch_size = 64  # Larger batches
   dropout_rate = 0.2  # More regularization
   ```

5. **Additional Regularization:**
   ```python
   Dense(128, activation='relu', kernel_regularizer=l2(0.001))
   EarlyStopping(monitor='val_loss', patience=5)
   ReduceLROnPlateau(factor=0.5, patience=3)
   ```

6. **Different Architectures:**
   - ResNet: Residual networks (better for deep models)
   - MobileNet: Lightweight (mobile deployment)
   - Inception: Complex feature extraction
   - EfficientNet: Balanced accuracy & efficiency

---

## Learning Resources

- **TensorFlow/Keras:** https://www.tensorflow.org/guide
- **CNN Basics:** https://cs231n.github.io/
- **Image Processing:** https://docs.opencv.org/
- **NumPy:** https://numpy.org/doc/
- **Matplotlib:** https://matplotlib.org/stable/contents.html

---

## Key Concepts Explained

### **Convolution Operation:**
```
3×3 Filter scans image:
┌───┐
│ # │ ← Kernel
└───┘

Output = Sum of (Filter element × Image pixel)
Repeated across entire image = Feature Map
```

### **ReLU Activation:**
```
ReLU(x) = max(0, x)

Input:  [-2, -1, 0, 1, 2, 3]
Output: [0,  0,  0, 1, 2, 3]  ← Negative values become 0

Purpose: Introduce non-linearity
Allows learning of complex patterns
```

### **Sigmoid Activation:**
```
Sigmoid(x) = 1 / (1 + e^-x)

Output range: [0, 1]
0 → Cat, 1 → Dog
0.5 → Uncertain
```

### **Batch Normalization:**
```
Before: [100, 200, 150, 180]  ← Large values
After:  [0.5, 1.2, 0.1, 0.8]  ← Normalized (-mean)/std

Benefits:
├─ Faster training
├─ Stable gradients
└─ Regularization effect
```

---

## Contributing

Feel free to:
- Improve model architecture
- Add data augmentation
- Implement transfer learning
- Optimize hyperparameters
- Fix any issues

---

## License

This project uses the Kaggle public dataset. Refer to Kaggle's terms for dataset usage.

---

## Support

For questions about:
- **CNN Architecture:** Refer to TensorFlow documentation
- **Image Processing:** Check OpenCV docs
- **Dataset Issues:** See Kaggle dataset page
- **Training Problems:** See TensorFlow debugging guide

---

## Summary

- This project demonstrates:
- Building CNN from scratch  
- Image preprocessing and normalization  
- Convolutional feature extraction  
- Handling overfitting with batch norm & dropout  
- Binary image classification  
- Performance visualization  
- Making predictions on new images  

---
**Status:** Complete and Ready to Use
