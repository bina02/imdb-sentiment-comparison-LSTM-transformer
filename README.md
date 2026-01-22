# 🚀 Performance Comparison: LSTM vs. Transformer on IMDB Dataset

> This project compares the performance of **RNN, LSTM, and Transformer** models for sentiment classification, specifically analyzing the impact of model depth and architectural robustness.

## 📌 1. Key Performance Metrics

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
| | **Transformer(lr:0.0001)**| **6** | **0.9138** | **0.8481** | **0.9373** |

## 🛠 2. Tech Stack
* **Framework**: PyTorch
* **Tokenizer**: SentencePiece (Subword Tokenization)
* **Library**: Scikit-learn (Metrics), Matplotlib (Visualization)
* **Optimization**: Adam Optimizer (LR 1e-4), Dropout (0.3)

## 📊 3. Data & Model Architecture
* **Dataset**: IMDB Movie Reviews (Binary Sentiment Classification)
* **Model Structures**: 
  - **Simple RNN**: Vanilla Recurrent Neural Network.
  - **LSTM**: Long Short-Term Memory capturing long-term dependencies.
  - **Transformer**: Encoder-only architecture focusing on Self-Attention.

## 📉 4. Analysis

###  Beyond Accuracy: The Power of ROC AUC
Accuracy can be misleading on sensitive thresholds. **ROC AUC** proves the models' underlying discriminative power. Phase 2 results show both LSTM and Transformer achieving AUC > 0.93, indicating superior ability to distinguish sentiment.


###  LSTM: Peak Performance in Shallow Layers
LSTM hit **99.8% Training Accuracy** in Phase 1, showing exceptional memorization. The **2-Layer LSTM** achieved the best test score (0.8798), making it the optimal structure for this data scale.

###  Transformer: Structural Stability in Deep Nets
In Phase 3, RNN/LSTM suffered from **Model Collapse** (Acc 0.50~0.57). However, the **Transformer (6-Layers)** maintained a high ROC AUC of 0.9373, highlighting the effectiveness of **Residual Connections**.


## ⚠️ 5. Failed Attempts & Troubleshooting

* **Vanishing Gradient in Deep RNNs**: 
    Initially, I attempted to improve performance by simply adding more layers to the RNN and LSTM (Phase 3). However, this led to a massive drop in accuracy (~50%). I realized that without residual connections, gradients vanish in deeper recurrent layers, making them unable to learn even the training data.
* **Initial Overfitting of LSTM**: 
    The 1-Layer LSTM showed a Train Acc of 99.8% but a lower Test Acc. I tried increasing Dropout and reducing the Learning Rate in Phase 2, which successfully narrowed the gap and improved generalization.
* **Transformer Tuning**: 
    The Transformer was sensitive to the Learning Rate. Setting it too high caused the loss to diverge, which taught me the importance of a stable LR (1e-4) for attention-based models.

## 🧠 6. Lessons Learned

1.  **"Deeper" is not always "Better"**: Adding complexity without structural support (like Skip Connections) can destroy a model's ability to converge.
2.  **Metrics Matter**: Relying only on Accuracy can be dangerous. Comparing it with ROC AUC helped me identify whether a model was actually learning or just guessing.
3.  **Architectural Strengths**: I learned firsthand why Transformers dominate the industry today—their ability to scale without collapsing is their greatest strength.

## 🏁 7. Conclusion
* **Optimal Model**: **2-Layer LSTM** is the most efficient choice for IMDB-sized data.
* **Robustness**: **Transformer** is the most stable as it scales deeper.
* **Key Takeaway**: Recurrent models have a clear "depth ceiling," while Transformers excel in deep learning tasks.

## 🏃 8. Getting Started

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git](https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git)
   cd imdb-sentiment-comparison-LSTM-transformer
