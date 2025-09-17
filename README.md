# Sign_mnist_classification
Project focuses on building a Convolutional Neural Network (CNN) to classify hand gestures from images using the Sign Language MNIST dataset. The dataset consists of images of American Sign Language (ASL) gestures, where each image is represented as a flattened vector of pixel values along with a corresponding label.
# CNN-based Sign Language Recognition

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify hand signs using the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist). The model is built with TensorFlow and trained to recognize 24 different hand gestures.

## 📂 Dataset
- `sign_mnist_train.csv`: Contains training images of hand signs represented as flattened pixel values along with their labels.
- `test1.png`: Example image for testing the model after training.

## 🧰 Libraries Used
- `numpy`, `pandas` for data manipulation
- `matplotlib` for visualization
- `tensorflow` for deep learning
- `sklearn` for data splitting
- `PIL` for image processing

## 🚀 How to Run
1. Clone the repository.
2. Ensure the dataset files are in the working directory.
3. Run the Jupyter notebook `CNN_Deep_Learning.ipynb`.
4. Visualize the data and train the CNN model.
5. Test the model using the provided image or your own images.

## 📈 Model Architecture
- Two convolutional layers with ReLU activation.
- Max pooling layers to reduce spatial dimensions.
- Flatten layer to convert 2D data into 1D.
- Three dense layers with 800, 600, and 400 neurons.
- Final output layer with softmax activation for classification.

## 📊 Results
- Training and validation accuracy are tracked across epochs.
- Final evaluation is performed on the test set.
- The model can predict new hand sign images after resizing and preprocessing.

## ✅ Suggestions for Improvement
- Tune hyperparameters like learning rate, batch size, and number of epochs.
- Implement data augmentation to improve generalization.
- Experiment with dropout or batch normalization to reduce overfitting.
- Expand the dataset to include more variations and noise.
- Use advanced architectures like transfer learning for better accuracy.
