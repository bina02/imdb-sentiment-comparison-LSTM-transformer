# 🚀 Performance Comparison: LSTM vs. Transformer on IMDB Dataset

> This project conducts a comprehensive comparative analysis of **RNN, LSTM, and Transformer** models for sentiment classification. It explores the impact of model depth, architectural robustness, and the relationship between evaluation metrics.

---

## 📌 1. Experimental Environment & Hyperparameters

To ensure a fair comparison, all models were trained under a unified global configuration with specific adjustments made in later phases to optimize performance.

### **Common Configuration**
* **Max Sequence Length**: 512
* **Embedding Dimension**: 128
* **Hidden Size (RNN/LSTM)**: 128
* **Transformer d_model**: 128
* **Transformer d_ff (Feed-forward)**: 512
* **Number of Heads**: 8
* **Dropout Rate**: 0.3 (Applied from Phase 2)
* **Tokenizer**: SentencePiece (Subword Tokenization)

### **Integrated Performance Metrics**

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

---

## 📊 2. Model Architecture
Based on the implementation in `models.ipynb`, the following architectures were utilized:

* **Simple RNN**: Uses the hidden state of the final time step (`h[-1]`) for classification.
* **LSTM**: Captures long-term dependencies using gated mechanisms and utilizes `pack_padded_sequence` for efficient variable-length sequence processing.
* **Transformer (Encoder-only)**: Features custom `MultiHeadAttention` and `EncoderLayer`.
    * **Positional Encoding**: Sinusoidal encoding is applied to provide sequence order information.
    * **Global Average Pooling**: Instead of a [CLS] token, the output is averaged (`x.mean(dim=1)`) across the sequence length before the final linear classifier.



---

## 📈 3. Analysis of Experimental Phases

### **Phase 1: Baseline (1 Layer, LR=0.001)**
Initially, I expected the Transformer to show superior performance due to its attention mechanism. However, **LSTM unexpectedly delivered the best results**. I observed a significant gap between Train and Test performance in all models, suggesting a high degree of overfitting. I hypothesized that shallow models (1-layer) might lack sufficient context understanding for complex sentiment.

### **Phase 2: Optimization (2 Layers, Transformer LR=0.0001, Dropout=0.3)**
To improve sentence comprehension, I increased the depth to 2 layers and added a 0.3 dropout rate. I specifically lowered the **Transformer's learning rate to 0.0001** to prevent divergence and ensure stable convergence.
* **Result**: LSTM achieved the project's highest score (**Test Acc 0.8798 / ROC AUC 0.9498**).
* **Insight**: Simple RNN's performance began to decline. This indicates that even at 2 layers, vanilla RNNs start suffering from the **Vanishing Gradient** problem.

### **Phase 3: Deep Architecture Stability (RNN/LSTM=4 layers, Transformer=6 layers)**
I pushed the limits of each architecture to test stability in deep configurations.
* **Result**: RNN and LSTM performance **collapsed** (Accuracy ~0.50), indicating they failed to learn as depth increased.
* **Insight**: The **Transformer remained remarkably stable** even with 6 layers, maintaining an ROC AUC of 0.9373. This is a direct result of the **Add & Norm (Residual Connections)** and Layer Normalization implemented in the `EncoderLayer`, which mitigates vanishing gradients.
* **Conclusion**: Despite the Transformer's stability, it did not surpass the 2-layer LSTM. This suggests that for the IMDB dataset, context captured within a `max_len=512` window may be more efficiently processed by an optimized LSTM than a deep Transformer.



---

## 🧠 4. Challenges & Lessons Learned

* **Hardware Constraints**: GPU memory limits forced a cap on `max_len` at 512. This likely limited the Transformer's ability to utilize long-range global dependencies which is its primary strength.
* **Hyperparameter Sensitivity**: The Transformer was extremely sensitive to the learning rate. Finding the balance between 0.001 and 0.0001 was crucial for performance.
* **The "Depth Ceiling"**: I confirmed that recurrent models have a clear structural ceiling. Without residual connections, adding layers to an LSTM/RNN eventually leads to total learning failure.

---

## 🏃 5. Getting Started

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git](https://github.com/bina02/imdb-sentiment-comparison-LSTM-transformer.git)
   cd imdb-sentiment-comparison-LSTM-transformer
