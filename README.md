
# BABY GPT 

LLM inspired by Andrej Karparthy and KellerJordan.

## ARCHITECTURE 

![Architecture of Kaparthy LLM](https://github.com/user-attachments/assets/b76e595b-6396-4754-81c5-0f1eb6c09b61)

Architecture is based on GPT2 as described by Andrej Kaparthy. 

DATA
The model is trained on the Hugging Face dataset Fineweb 10B samples. To eliminate cyclic behavior encountered by Mr. Karpathy, I've developed a randomization process that allows training on chaining the order of data.
![cyclic behavior during training on FineWeb ](https://github.com/user-attachments/assets/57291b0c-1adc-41f3-9215-b329839a079d)

Chunk data by EOF token. For performance-related issues, search for the EOF token only around the cursor (cursor jump 30M tokens) in range (±20k tokens, increased iteratively if there is no EOF in the range).
![Chanking](https://github.com/user-attachments/assets/76f7ec79-e801-4ec6-814e-74ba5a0afa96)
Chunked data can be randomly shuffled.
Chunks are streamlined into data streams for effortless use in multi-GPU training.
![Data Streams](https://github.com/user-attachments/assets/45fd8109-6356-44dc-a421-2cca32dff8c8)

The last documents in every stream are temporarily shortened to fit into the gradient accumulation batch_size * block_size perfectly.


# HELLA SWAG EVAL

- "ctx": "A man is sitting on a roof. he",
- "label": 3,
- "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."]
  
Every X iterations during training, Hella Swag evaluation is performed by predicting the next token for each sentence word-by-word. The winning answer is the one with the lowest overall summed loss.


## To do

* KV-cache
* Rotatory positional embedings
* QK-Normalization
* Final Training
## Training


```bash
  pip install -r requirements.txt
  ./fineweb.py
  torchrun --standalone --nproc_per_node=4 train.py
```
    
## Acknowledgements

 - [Andrej Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
 - [KellerJordan modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)


