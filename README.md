# Building-a-PyTorch-Model-to-Classify-Playing-Cards

This project demonstrates how to build an image classification model using **PyTorch** to recognize 53 types of playing cards. We leverage **transfer learning** with the **EfficientNet-B0** model for high accuracy. The project walks through data preparation, model training, evaluation, and visualization.

---

## 📂 Dataset Description

- **Image Format**: JPG, 224x224x3 (RGB)
- **Size**:
  - **Training**: 7,624 images
  - **Validation**: 265 images
  - **Test**: 265 images
- **Classes**: 53 (e.g., Ace of Clubs, King of Hearts, Joker)
- **Structure**:
  ```
  /dataset
      /train
          /ace of clubs
          /ace of diamonds
          ...
      /valid
      /test
      labels.csv
  ```

---

## 🚀 Project Workflow

1. **Setup and Imports**  
   Import required libraries: PyTorch, torchvision, timm (for pre-trained models), and others for visualization and data handling.

2. **Data Loading and Preprocessing**  
   - Create a custom `PlayingCardDataset` class using PyTorch's `Dataset`.
   - Apply transformations (resize, normalize) using `torchvision.transforms`.
   - Load data using `DataLoader` for efficient batching.

3. **Model Building**  
   - Use the pre-trained **EfficientNet-B0** model from `timm`.
   - Replace the final layer to classify 53 card types.

4. **Training the Model**  
   - Define the **loss function** (`CrossEntropyLoss`) and **optimizer** (`Adam`).
   - Implement a training loop with forward propagation, backpropagation, and validation.

5. **Evaluation and Visualization**  
   - Evaluate model performance on the test set.
   - Visualize **loss** and **accuracy** over epochs.
   - Display sample predictions with corresponding probabilities.

---

## 📊 Results

- **Final Test Accuracy**: **96.6%**
- The model successfully identifies different playing cards with high precision.
- Visualizations of training and validation loss/accuracy indicate good model performance with minimal overfitting.

---

## 🔑 Key Learnings

- **PyTorch Datasets and DataLoaders**: Efficient data handling for training and validation.
- **Transfer Learning**: Leveraging pre-trained models to improve accuracy and reduce training time.
- **Model Training**: Implementing custom training loops with loss calculation and backpropagation.
- **Visualization**: Tracking model performance using plots and visualizing predictions.

---

## 📦 How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/playing-card-classifier.git
   cd playing-card-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Playing_Card_Classification.ipynb
   ```

---

## 📌 Future Improvements

- Implement **data augmentation** to improve generalization.
- Experiment with **different architectures** (e.g., ResNet, ViT).
- Perform **hyperparameter tuning** for better performance.

---
