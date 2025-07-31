# autoencoder_for_mnist_dataset

# 🧥 Fashion MNIST - Autoencoder Clustering & Visualization

This project explores unsupervised learning on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using stacked autoencoders. The goal is to compress high-dimensional image data, cluster similar items, and visualize meaningful structure in an unlabeled setting.

---

## 📂 Project Structure



---

## 📊 Objective

- Build a stacked autoencoder using **SELU** activation.
- Train it to reconstruct Fashion MNIST images.
- Use **encoder outputs** as compressed representations.
- Apply **KMeans clustering** on encoded data.
- Visualize clusters using **t-SNE / UMAP**.
- Display real images from each cluster.

---

## 🛠️ Technologies

- Python 🐍
- TensorFlow / Keras
- scikit-learn
- Matplotlib, Seaborn
- t-SNE / UMAP

---

## 🧠 Model Architecture

```python
Encoder:
  Input (28x28)
  → Flatten
  → Dense(128, activation='selu')
  → Dense(64, activation='selu')
  → Dense(32, activation='selu')  # Bottleneck

Decoder:
  → Dense(64, activation='selu')
  → Dense(128, activation='selu')
  → Dense(784, activation='sigmoid')
  → Reshape(28x28)
