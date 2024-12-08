# AlzDetect
![GitHub Issues](https://img.shields.io/github/issues/shvm-k/AlzDetect)
![GitHub Forks](https://img.shields.io/github/forks/shvm-k/AlzDetect)
![GitHub Stars](https://img.shields.io/github/stars/shvm-k/AlzDetect)

Alzheimer's disease is a progressive neurological disorder that impacts memory, thinking, and behavior. Early and accurate diagnosis is critical for effective treatment and management. This project utilizes a pre-trained deep learning model to classify MRI scans into four categories:  
- **Non-Demented**  
- **Mild Demented**  
- **Moderate Demented**  
- **Very Mild Demented**  

The goal is to assist in early diagnosis using MRI data, thus contributing to the healthcare domain.

---

### 📂 **Dataset Overview**

The dataset contains MRI brain scans labeled into the above four categories. Key details about the dataset:  
- **Source**: Publicly available dataset [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis).  
- **Preprocessing**: Images are resized to a uniform dimension, normalized, and their labels are one-hot encoded.  
- **Split**: Divided into training, validation, and test sets.

---

### 🛠️ **Implementation**

#### **Preprocessing**  
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
  <img src="https://github.com/user-attachments/assets/477ef488-852c-4636-9cb1-b9d038164dba" alt="Augmented Training Data Samples" width="500" />
</p>

---

### 🧠 **Model Architecture**  
The project uses **EfficientNetB0** as the base model, fine-tuned for Alzheimer's classification. Key layers added include:  
- **Global Average Pooling** for feature extraction.  
- **Dense Layers** with ReLU activation for classification.  
- **Softmax Output Layer** for predicting probabilities across four classes.

---

### 🏋️ **Training**  
The model is trained using the **categorical crossentropy loss** function with the Adam optimizer. Early stopping and learning rate reduction techniques are employed to avoid overfitting and improve convergence.

---

### 🐞 **Known Issues**  
- [ ] Error in `model.fit`: Target and output shape mismatch.  
- [ ] Fuzzy logic not yet implemented in the evaluation phase.
 <img width="300" alt="Screenshot 2024-12-08 at 12 18 00 PM" src="https://github.com/user-attachments/assets/65107c9c-5c54-4760-b928-689b1dbde8fc">

---

## 🤝 **Contributing**  
We welcome contributions to improve the project and resolve issues. Here’s how you can help:

1. **Fork the repository**  
   - Go to the repository page.  
   - Click the **"Fork"** button in the top-right corner.

2. **Clone your fork**  


3. **Create a branch**  

4. **Make changes and commit**  
- Fix issues or add features.  
- Stage your changes:  
  ```
  git add .
  ```
- Commit your changes:  
  ```
  git commit -m "Fix: Error in model.fit"
  ```

5. **Push changes**  
- Push your branch:  
  ```
  git push origin fix-error-xyz
  ```

6. **Submit a pull request (PR)**  
- Open a pull request on the original repository.

---

### 🚀 **Future Work**  

- **Fuzzy Logic Integration**: Incorporate fuzzy rules for enhanced classification under uncertainty.  
- **Deployment**: Package the model into a web application for healthcare practitioners.  
- **Explainability**: Use Grad-CAM to provide insights into the regions of the brain contributing to classification.  
- **Larger Dataset**: Extend training to a more diverse dataset for improved generalizability.


### 🛠️ **Open Source Collaboration**  
This project is open source and relies on contributions from the community to address issues, add features, and improve overall quality. Please report bugs, suggest features, or resolve existing issues in the **[Issues](./issues)** tab.



