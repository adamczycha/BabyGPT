
# BabyGPT

BabyGPT is a GPT-2 (127M parameter) model built on the foundations of NanoGPT by Andrej Karpathy. This project adapts the original NanoGPT approach for Polish language data and introduces several custom improvements to the data handling and training process.

---

## Overview

- **Model Architecture:**  
  Based on GPT-2 (127M).  
- **Training Data:**  
  Trained on a Polish subset of the fineweb2 dataset, with evaluations performed on a version of HellaSwag translated by openGPT-X.
- **Training Hardware:**  
  Training was carried out on 8× A40 GPUs over a period of 5 hours.
- **Observations:**  
  The exceptionally low training and validation loss might be partly due to the tokenization strategy not being fully optimized for the Polish language. In hindsight, a multilingual tokenizer might have offered better compression and performance. Additionally, using A40 GPUs was chosen for cost reasons, although A100 or H100 cards could have achieved similar or better results in less time and probably lower cost :*)...

![Training run plots](https://github.com/user-attachments/assets/2d227e5c-dd4d-43a1-bf3d-8e42935216a5)



## Key Modifications Compared to NanoGPT


### Dataset
- The training utilizes the [Polish chank of Fineweb2 ](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) prepared by Hugging Face. The data was tokenized using the GPT-2 tokenizer, which led to an inflation in the number of tokens and a decrease in the amount of information per token. Unsurprisingly in hindsight, this issue was not checked after tokenization and somehow went unnoticed until results of training process.


### Preprocessing Steps
1. **Tokenization & Chunking:**  
   The raw text is tokenized and then segmented based on the detection of EOF tokens. The algorithm searches within a window of roughly ±20k tokens to efficiently determine chunk boundaries.  
   ![Searching for EOF with Cursor](https://github.com/user-attachments/assets/ae086ce9-3706-4b72-9a21-546dca585bbb)

2. **Random Shuffling:**  
   To prevent cyclic patterns in traning loss, the chunks are randomly rearranged before being stitched together. 
   ![Cyclic Behaviour of FineWeb 10B](https://github.com/user-attachments/assets/42c3391f-cd66-47b1-97d8-c6eb7ce14f86)
   Above behaviour was observered by Andrej Karpathy during traning on FineEdu 10B.  

3. **Streamlining:**  
   The shuffled chunks are then concatenated into continuous streams for each GPU.  
   ![Data Streams](https://github.com/user-attachments/assets/45fd8109-6356-44dc-a421-2cca32dff8c8)

### Evaluation
BabyGPT predicts the next token in a sequence, word by word, in each of four cases. Among these predictions, the one with the lowest average loss is selected as the chosen answer and compared with the labels. For each GPU, there are mini_batch / 4 samples.

**Example Prediction:**
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



## Planned Improvements

- Replace the current tokenizer with a multilingual variant.
- Filter and refine the Polish fineweb2 dataset to create a high-quality "fineedu" subset.
- Implement key-value (kv) cache for faster inference.
- Introduce rotary positional embeddings.
- Experiment with ReLU² activation functions.
- Integrate QK-Normalization for enhanced model stability.



## Getting Started

### Installation

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Training Script

1. **Single GPU Training:**
   ```bash
   python train.py
   ```

2. **Multi-GPU Training:**
   ```bash
   torchrun --standalone --nproc_per_node=4 train.py
   ```

> **Note:**  
> Before starting the training, update the `train_config.py` file. Set the folder name for your training data in the appropriate field and adjust the `mini_batch` size to match your GPU's memory capacity.

The `train_config.py` file also allows you to modify key parameters:
- **general:** Choose modes (sampling, evaluation, training, or validation) or combine them.
- **data:** Select the appropriate dataset for training and evaluation.
- **training, validation, sampling, evaluation:** Change details as needed.



## Acknowledgements

Special thanks to:
- **Andrej Karpathy** – for NanoGPT, which laid the groundwork for this project.
- **KellerJordan** – for contributions through modded-nanogpt, advancing efficient training methods.
- 


