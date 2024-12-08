# AlzDetect

Alzheimer's disease is a progressive neurological disorder that impacts memory, thinking, and behavior. Early and accurate diagnosis is critical for effective treatment and management. This project utilizes a pre-trained deep learning model to classify MRI scans into four categories:
- **Non-Demented**
- **Mild Demented**
- **Moderate Demented**
- **Very Mild Demented**

The goal is to assist in early diagnosis using MRI data, thus contributing to the healthcare domain.


#### 📂 **Dataset Overview**

The dataset contains MRI brain scans labeled into the above four categories. Key details about the dataset:  
- **Source**: Publicly available dataset (e.g., Kaggle or another relevant source).  
- **Preprocessing**: Images are resized to a uniform dimension, normalized, and their labels are one-hot encoded.  
- **Split**: Divided into training, validation, and test sets.

#### 🛠️ **Implementation**

##### **Preprocessing**
- Images are resized to 300x300 pixels.
- Pixel values are normalized to improve model convergence.
- Labels are one-hot encoded for multi-class classification.

#### **Data Augmentation**
To enhance the model's ability to generalize, the following augmentations are applied:
- Random rotation
- Zoom
- Horizontal and vertical flipping
- Brightness variations

Below is an example of augmented training data:
<p>
  <img src="https://github.com/user-attachments/assets/477ef488-852c-4636-9cb1-b9d038164dba" alt="Augmented Training Data Samples" width="300" />
</p>

#### **Model Architecture**
The project uses **EfficientNetB0** as the base model, fine-tuned for Alzheimer's classification. Key layers added include:
- **Global Average Pooling** for feature extraction.
- **Dense Layers** with ReLU activation for classification.
- **Softmax Output Layer** for predicting probabilities across four classes.

### **Training**
The model is trained using the **categorical crossentropy loss** function with the Adam optimizer. Early stopping and learning rate reduction techniques are employed to avoid overfitting and improve convergence.

---
### 🚀 **Future Work**

- **Fuzzy Logic Integration**: To incorporate fuzzy rules for enhanced classification under uncertainty.  
- **Deployment**: Package the model into a web application for healthcare practitioners.  
- **Explainability**: Use Grad-CAM to provide insights into the regions of the brain contributing to classification.  
- **Larger Dataset**: Extend training to a more diverse dataset for improved generalizability.  
