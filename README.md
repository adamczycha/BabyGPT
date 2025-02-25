# BABY GPT

Welcome to BABY GPT – a fun, lightweight large language model inspired by two cool projects: Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) and KellerJordan’s [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). This project takes the best parts of both worlds: the simplicity of nanoGPT and the speed optimizations of modded-nanogpt, all while keeping things accessible and easy to tinker with.

---

## Table of Contents

- [What is BABY GPT?](#what-is-baby-gpt)
- [Where Did It Come From?](#where-did-it-come-from)
- [How It Works](#how-it-works)
- [Handling the Data](#handling-the-data)
- [Training and Evaluation](#training-and-evaluation)
- [What's Next?](#whats-next)
- [Thanks and Acknowledgements](#thanks-and-acknowledgements)
- [License](#license)

---

## What is BABY GPT?

BABY GPT is my friendly little language model built on the classic GPT-2 design. It’s designed to be simple enough to play around with, yet powerful enough to deliver some neat results.
---

## Where Did It Come From?

This project started as a blend of two awesome ideas:

- **nanoGPT by Andrej Karpathy:**  
  A super minimalist and easy-to-read repository that’s great for learning and quick experiments.

- **modded-nanogpt by KellerJordan:**  
  An advanced, speed-optimized version that shows what’s possible when you push your hardware to the limit.

I took inspiration from both, aiming to create a model that’s straightforward to use while still packing a punch in performance.

---

## How It Works

### The Model
BABY GPT follows the GPT-2 blueprint with a few smart tweaks:

- **GPT-2 Basics:**  
  It uses a Transformer architecture with multi-head self-attention and feedforward layers—pretty much the standard stuff that makes GPT-2 tick.

- **My Tweaks:**  
  - **Data Chunking:**  
    Split up the training data by looking for EOF tokens. This helps break up the text into manageable chunks and stops the model from getting stuck in repetitive loops.
  - **Streamlined Data:**  
    The chunks get shuffled and stitched together into neat streams, making it easier to train on multiple GPUs without any hiccups.

---

## Handling the Data

- **Dataset:**  
  I use the [Fineweb 10B dataset](https://huggingface.co/datasets) from Hugging Face. It’s packed with diverse language samples perfect for training.

- **Preprocessing:**  
  1. **Tokenization & Chunking:**  
     The text is tokenized and then split based on EOF tokens. We check for these tokens around the current spot (roughly ±20k tokens) to keep things efficient.
     ![searching for EOF with cursor](https://github.com/user-attachments/assets/ae086ce9-3706-4b72-9a21-546dca585bbb)


  3. **Random Shuffling:**  
     The chunks are randomly rearranged to avoid any predictable, cyclic patterns.
     ![cyclic behaviour of FineWeb 10b](https://github.com/user-attachments/assets/42c3391f-cd66-47b1-97d8-c6eb7ce14f86)

  5. **Streamlining:**  
     Finally, the chunks are combined into continuous streams that fit perfectly into training batches.
     ![Data Streams](https://github.com/user-attachments/assets/45fd8109-6356-44dc-a421-2cca32dff8c8)

---

## Training and Evaluation

### Getting Started

1. **Install Dependencies:**  
   Just run:

   ```bash
   pip install -r requirements.txt
   ```

2. **Training on Multiple GPUs:**  
   If you have a multi-GPU setup, try this:

   ```bash
   torchrun --standalone --nproc_per_node=4 train.py
   ```

3. **Training on a Single GPU or CPU:**  
   No worries if you’re on a less powerful machine. Adjust the parameters like this:

   ```bash
   python ./train.py
   ```

### Hella Swag Evaluation

Every so often, BABY GPT takes a break to see how well it’s learning using the Hella Swag benchmark:

- **What’s Happening:**  
  The model predicts the next token in a sentence one word at a time. Out of several possible endings, the one with the lowest overall loss wins.
- **Example:**  

  ```json
  {
    "ctx": "A man is sitting on a roof. he",
    "label": 3,
    "endings": [
      "is using wrap to wrap a pair of skis.",
      "is ripping level tiles off.",
      "is holding a rubik's cube.",
      "starts pulling up roofing on a roof."
    ]
  }
  ```

This helps us check that the model isn’t just memorizing but is actually understanding language.

---

## What's Next?

There’s always room for improvement! Here are a few things on my to-do list:

- **KV-Cache:** Adding caching for even faster inference.
- **Rotary Positional Embeddings:** To give the model a better sense of word order.
- **QK-Normalization:** For more stable and efficient attention.
- **Fine-Tuning:** More tweaks to the hyperparameters to squeeze out every bit of performance.

---

## Thanks and Acknowledgements

Big thanks to:
- **Andrej Karpathy** for nanoGPT, which laid the groundwork for many of us.
- **KellerJordan** for modded-nanogpt, pushing the limits of what’s possible with fast training.
- And everyone in the open-source community for continuously inspiring and sharing their work.
