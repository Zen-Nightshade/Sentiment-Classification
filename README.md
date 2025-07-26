# Sentiment Classification with RNNs, LSTMs and Attention Mechanisms

## Project Overview

This repository explores binary sentiment classification  using various RNN and LSTM architectures, focusing on how attention mechanisms can improve model performance. The research investigates which neural architecture is most effective for classifying text as expressing either positive or negative sentiment.

## Architecture Overview and Attention Integration

To efficiently explore 16 model variants, this project uses a **modular design** with just 4 core classes:
- `AdditiveAttention`
- `MultiplicativeAttention`
- `AttentiveRNN`
- `AttentiveLSTM`

These classes are flexibly combined to construct all 16 model variants, covering:
- Uni-directional and bi-directional RNNs
- Uni-directional and bi-directional LSTMs
- Each of the above with and without additive, multiplicative, and concatenative attention

This approach avoids redundant code and enables rapid experimentation with different architectures and attention mechanisms.

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy**
- **Precision, Recall, F1-score** (micro and macro averages)
- **Confusion Matrix**

Comprehensive metrics are logged for each experiment to enable detailed performance comparison.

## Key Findings

- **Best Model:** The BiLSTM with concatenative attention achieved the highest test accuracy of **89.12%**, highlighting the synergy of bidirectional memory and attention for sentiment classification.
- **Interpretability:** A custom heatmap visualizer was developed to interpret attention weights across all 16 models, offering insights into which parts of the text the models focus on during prediction.

## Features

- Modular implementation: only 4 core classes for all variants
- Automated evaluation and metric logging
- Integrated heatmap visualization for attention weights

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/Zen-Nightshade/Sentiment-Classification.git
    cd Sentiment-Classification
    ```
2. Set up the environment and install dependencies as per the `requirements.txt` file.

3. Run experiments using the provided scripts/notebooks.

## Visualizations

The repository includes tools to visualize attention weights, helping to interpret how models make decisions based on input text.

## Results

| Model                           | Accuracy (%) |
|----------------------------------|--------------|
| BiLSTM + Concatenative Attention | 89.12        |
| Other models                    | See logs     |

## Reference

For a detailed description of the experiments, results, and visualizations, please refer to [Report.pdf](https://github.com/Zen-Nightshade/Sentiment-Classification/blob/main/Report.pdf).

---

**Author:** [Zen-Nightshade](https://github.com/Zen-Nightshade)
