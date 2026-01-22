# 🚀 Performance Comparison: LSTM vs. Transformer on IMDB Dataset

> This project compares the performance of **RNN, LSTM, and Transformer** models for sentiment classification, specifically analyzing the impact of model depth and architectural robustness.

## 📌 1. Key Performance Metrics

The results show the progression from shallow baseline models to deeper architectures across three experimental phases.

| Phase | Model | Layer | Train Acc | Test Acc | Test ROC AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 1** | RNN | 1 | 0.8378 | 0.7123 | 0.7745 |
| (Baseline) | **LSTM** | 1 | 0.9980 | 0.8630 | 0.9296 |
| | Transformer | 1 | 0.9505 | 0.8474 | 0.9257 |
| **Phase 2** | RNN | 2 | 0.7603 | 0.6416 | 0.6965 |
| (Optimized)| **LSTM (SOTA)** | **2** | **0.9864** | **0.8798** | **0.9498** |
| | **Transformer**| 2 | 0.9193 | 0.8582 | 0.9355 |
| **Phase 3** | RNN | 4 | 0.5009 | 0.5043 | 0.5053 |
| (Deep) | LSTM | 4 | 0.6033 | 0.5715 | 0.6143 |
| | **Transformer**| **6** | **0.9138** | **0.8481** | **0.9373** |

## 🛠 2. Tech Stack
* **Framework**: PyTorch
* **Tokenizer**: SentencePiece (Subword Tokenization)
* **Library**: Scikit-learn (Metrics), Matplotlib (Visualization)
* **Optimization**: Adam Optimizer (LR 1e-4), Dropout (0.3)

## 📊 3. Data & Model Architecture
* **Dataset**: IMDB Movie Reviews (Binary Sentiment Classification)
* **Model Structures**: 
  - **Simple RNN**: Vanilla Recurrent Neural Network.
  - **LSTM**: Long Short-Term Memory with gate mechanisms to capture long-term dependencies.
  - **Transformer**: Encoder-only architecture focusing on Self-Attention mechanisms.


## 📈 4. Analyze

###  Beyond Accuracy: The Power of ROC AUC
While Accuracy measures the hit rate at a 0.5 threshold, **ROC AUC** proves the models' underlying discriminative power. Phase 2 results show both LSTM and Transformer achieving AUC > 0.93, indicating superior ability to distinguish sentiment regardless of the threshold.


###  LSTM: Peak Performance in Shallow Layers
LSTM demonstrated exceptional memorization, hitting **99.8% Training Accuracy** in Phase 1. The **2-Layer LSTM** achieved the overall best test score (0.8798), identifying it as the optimal structure for this specific data scale.

###  Transformer: Structural Stability in Deep Nets
As depth increased in Phase 3, RNN and LSTM suffered from **Model Collapse** (Test Acc 0.50~0.57) due to vanishing gradients. However, the **Transformer (6-Layers)** maintained a high ROC AUC of 0.9373. This highlights the effectiveness of **Residual Connections** and **Layer Normalization** in deep architectures.


## 🏁 5. Conclusion
* **Optimal Model**: For IMDB-sized datasets, a **2-Layer LSTM** is the most efficient and accurate choice.
* **Architectural Robustness**: **Transformer** is significantly more stable as the model scales deeper, showing the best generalization with the smallest gap between Train and Test metrics.
* **Key Takeaway**: Increasing layers in RNN-based models without proper structural support leads to learning failure, whereas Transformer scales robustly.

## 🏃 6. Getting Started
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git](https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git)
   cd imdb-sentiment-comparison-LSTM-transformer
