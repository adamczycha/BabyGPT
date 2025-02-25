
# BABY GPT 

LLM inspired by Andrej Karparthy and KellerJordan was built in the struggle to learn as much as possbile about LLM architecture and modern techniques of optimilization of learning. 


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


