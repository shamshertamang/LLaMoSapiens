# LLMaoSapiens  

This project is an **educational implementation of a GPT-style transformer Large Language Model** built completely from scratch in PyTorch.  
The primary goal was to **supplement and deepen my understanding of large language models (LLMs)**, inspired by the *“LLMs from Scratch”* book.  

Along the way, I also created a **custom dataset (~2 million tokens)** sourced from Yuval Noah Harari’s *Sapiens* trilogy, which served as the training corpus.  

⚠️ Note: This repository contains only implementation code and training scripts. Pre-trained model weights are not included. Use the experiment notebook to build your own llm.

---

## 📂 Repository Structure  

```text
LLMaoSapiens
│──GPTFromScratch/
│   │── architecture.py       # Defines the transformer block & GPT architecture
│   │── attention.py          # Implementation of self-attention mechanism
│   │── train.py              # Training loop & optimizer setup methods
│   │── experiment.ipynb      # Note for loading model, training and evaluation
│
│──TextDataProcessing/
│   │── DataPipeLine.ipynb    # experiment script for data loading and batching pipeline
│   │── PDF2TextFileConvert.ipynb # Conversion of Sapiens trilogy PDFs → raw text
│   │── data.py               # data loading and batching pipelines script
│
│──requirements.txt          # dependencies
│──README.md                 # Project overview
```

---

## 🚀 Explorations  

- Built a **GPT-style language model** from scratch:  
  - Tokenization, embeddings, positional encodings  
  - Multi-head self-attention and feed-forward layers  
  - Layer normalization, residual connections, dropout  
- **Custom dataset pipeline**: converted *Sapiens* trilogy PDFs → plain text → tokenized dataset (~2M tokens).  
- **Training utilities**: implemented from scratch with PyTorch, including batching, loss tracking, and experiment notebooks.  
- **Hands-on exploration** of scaling behaviors, model depth, and training stability.  

---

## 📊 Dataset  

- **Source**: *Sapiens*, *Homo Deus*, *21 Lessons for the 21st Century* by Yuval Noah Harari  
- **Size**: ~2 million tokens after preprocessing  

---

## 🛠️ Setup  

⚠️ Note: This repository contains only implementation code and training scripts. Pre-trained model weights are not included. To train the model, use the experiment notebook.

1. Clone the repo:  
   ```bash
   git clone https://github.com/<your-username>/GPTFromScratch.git
   cd GPTFromScratch
   ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
   ``` 
   
3. Run preprocessing and experiments:
- Convert raw PDFs → text: TextDataProcessing/PDF2TextFileConvert.ipynb
- Run the cells in the experiments.py under the GPTFromScratch folder to build dataset pipeline, build, train and evaluate the model.

---

## 🎯 Motivation

This project was built for fun and learning, to:

- Reinforce my understanding of transformer internals

- Explore dataset preparation and training dynamics

- Gain hands-on intuition for LLM scaling laws

It is not **intended as a production model**, but rather as a stepping stone toward deeper projects in ML and AI.

---

## 🤝 Acknowledgments

- LLMs From Scratch (book) for guidance and inspiration

- Yuval Noah Harari’s Sapiens trilogy for the dataset

- PyTorch community for robust ML tooling


