# Garbage Classification 
- Rust CNNA high-performance Convolutional Neural Network (CNN) for garbage classification built from scratch using **Rust** and the **tch** crate (PyTorch bindings).
- This project classifies waste into 6 categories: cardboard, glass, metal, paper, plastic, and trash with **75% validation accuracy** or more depending on how much you train it.

## 🎯 Project Overview- **Language**: Rust with tch (PyTorch C++ API bindings)
- **Model**: Custom CNN with batch normalization, dropout, and data augmentation
- **Classes**: 6 garbage types (cardboard, glass, metal, paper, plastic, trash)
- **Dataset Size**: 13,901 images
- **Performance**: 75% validation accuracy
- **Training Platform**: Kaggle GPUs

## 📊 Dataset**Source**: [Garbage Classification Dataset on Kaggle](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification)

- **Total Images**: 13,901
- **Training Set**: 11,120 images (80%)
- **Validation Set**: 2,781 images (20%)
- **Image Size**: 200x200 RGB
- **Data Augmentation**: Random horizontal/vertical flips, rotation, resizing

## 🚀 Features
### Model Architecture- **Feature Extraction**: 3 convolutional blocks (32→64→128 channels)
- **Pooling**: Max pooling and Global Average Pooling
- **Classification**: Fully connected layers with dropout (0.15)
- **Regularization**: Batch normalization, gradient clipping

### Training Features- **Learning Rate Scheduling**: StepLR with γ=0.85 every 5 epochs
- **Optimizer**: Adam with initial LR=1e-4
- **Loss Function**: Cross-entropy loss
- **Batch Size**: 32 (configurable)
- **Device Support**: Auto-detection of CUDA GPUs

### Evaluation & Visualization- **Comprehensive Metrics**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed misclassification analysis
- **Loss Curves**: Training/validation loss tracking
- **Export**: CSV files for external analysis + Python plotting scripts

## ⚙️ Installation & Setup### Prerequisites```bash
# Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### LibTorch Setup**For CPU-only (Linux):**
```bash
curl -O https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
```

**For GPU (Linux) with CUDA 12.1:**
```bash
curl -O https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
```
-unzip the zips

### Environment Variables (edit it to the right path)
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### Training Configurations| Config | Samples | Epochs | Device | Batch Size | Use Case |
```
|--------|---------|--------|--------|------------|----------|
| 1      | 500     | 2      | CPU    | 8          | Quick test |
| 2      | 2,000   | 5      | GPU    | 16         | Development |
| 3      | All     | 100    | GPU    | 32         | Full training |
| 4      | Custom  | Custom | Custom | Custom     | Experimentation |
```

## 📈 Results### Performance Metrics- **Overall Accuracy**: 75% (validation)
- **Training Time**: ~1 hour 23 minutes (100 epochs, GPU)
- **Final Training Loss**: 0.6566
- **Final Validation Loss**: 0.7176

### Per-Class Performance| Class     | Precision | Recall | F1-Score | Support |
```
|-----------|-----------|--------|----------|---------|
| Cardboard | 0.79      | 0.83   | 0.81     | 436     |
| Glass     | 0.67      | 0.68   | 0.67     | 440     |
| Metal     | 0.81      | 0.70   | 0.75     | 478     |
| Paper     | 0.74      | 0.72   | 0.73     | 506     |
| Plastic   | 0.72      | 0.77   | 0.75     | 500     |
| Trash     | 0.76      | 0.78   | 0.77     | 421     |
```

### Key Insights- **Best Performance**: Cardboard and Metal classification
- **Challenge Areas**: Glass classification (visually similar to other materials)
- **Common Confusions**: Glass ↔ Paper/Plastic (expected due to visual similarity)
- **Learning Curve**: Smooth convergence without overfitting

![Training Loss](training_validation_loss.png)
![Confusion Matrix](confusion_matrix.png)


## 🌐 Kaggle Integration**Live Demo**: [Garbage Classifier Rust on Kaggle](https://www.kaggle.com/code/aarymilindkinge/garbage-classifier-rust)

This project was developed and tested on **Kaggle Notebooks** using free GPU access, demonstrating the feasibility of Rust-based deep learning on cloud platforms.

### Kaggle Mock Setup
Check out this [link](https://www.kaggle.com/code/aarymilindkinge/garbage-classifier-rust) look for full example
```bash
# Clone repository
!git clone https://github.com/Not-Buddy/garbage_classification.git
%cd garbage_classification

# Install Rust + dependencies (handled in notebook)
# Run training
!cargo run --release -- 3
```
```
## 📁 Project Structure```
garbage_classification/
├── src/
│   ├── main.rs                 # Entry```int with CLI argument```rsing
│   ├── menu.rs                 # Configuration```nagement +```taset loading
│   ├── relu.rs                 # CNN```del architecture
│   ├── train_model.rs          # Training```op and optimization
│   ├── training_validation.rs  # Evaluation metrics an```lotting
│   └── visualize_data.rs       # Data loading and batch```sualization
├── Cargo.toml                  # Dependencies and project configuration
├── README.md                   # This file
└── plot_results.py             # Generated Python plotting script
```

## 🔧 Technical Implementation### Dataset Loading- **Parallel Processing**: Rayon for multi-threaded image loading
- **Augmentation Pipeline**: Random flips, rotation, resizing
- **Memory Efficiency**: Tensor operations in CHW format
- **Progress Tracking**: Real-time processing counter

### Model Architecture
```rust
CNN {
    features: Conv2d(32) -> BN -> ReLU -> Conv2d(32) -> BN -> ReLU -> MaxPool
              Conv2d(64) -> BN -> ReLU -> Conv2d(64) -> BN -> ReLU -> MaxPool  
              Conv2d(128) -> BN -> ReLU -> Conv2d(128) -> BN -> ReLU -> MaxPool,
    
    gap: GlobalAveragePooling2d,
    
    classifier: Linear(128) -> ReLU -> Dropout(0.15) -> Linear(6)
}
```

### Training Features- **Automatic Mixed Precision**: GPU optimization
- **Learning Rate Scheduling**: Exponential decay
- **Early Stopping**: Validation-based (configurable)
- **Gradient Clipping**: Prevents exploding gradients
- **Progress Monitoring**: Real-time loss and accuracy tracking

## 📊 Generated OutputsAfter training, the following files are automatically generated:

- `training_losses.csv` - Loss data for plotting
- `confusion_matrix.csv` - Confusion matrix data
- `evaluation_results.csv` - Detailed predictions with probabilities
- `training_stats.csv` - Complete training metrics
- `plot_results.py` - Python script for visualization
- `garbage_classifier_X_epochs_Y_samples.pt` - Saved model weights

## 🔄 Future Improvements### Model Enhancements- **Transfer Learning**: ResNet/EfficientNet backbones
- **Attention Mechanisms**: Focus on distinguishing features
- **Ensemble Methods**: Multiple model voting
- **Advanced Augmentation**: MixUp, CutMix, AutoAugment

### Engineering- **Early Stopping**: Automatic best model selection
- **Hyperparameter Tuning**: Automated search
- **Model Serving**: REST API for inference
- **Cross-Validation**: More robust evaluation

## 🏆 Why Rust?- **Performance**: Near-C++ speed with memory safety
- **Concurrency**: Excellent parallel processing support
- **Ecosystem**: Growing ML/DL ecosystem with tch
- **Deployment**: Single binary, no runtime dependencies
- **Reliability**: Compile-time guarantees prevent common ML bugs

## 🤝 Contributing1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 LicenseThis project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments- [tch crate](https://crates.io/crates/tch) for PyTorch bindings
- [Kaggle](https://kaggle.com) for free GPU access
- Dataset contributors for the garbage classification dataset
- Rust community for excellent documentation and support

***

**⭐ Star this repository if you found it helpful!**

For questions or collaboration opportunities, feel free to open an issue or reach out directly.

[1](https://www.kaggle.com/code/aarymilindkinge/garbage-classifier-rust)
