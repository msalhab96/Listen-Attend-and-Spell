# Listen, Attend and Spell (LAS)

This is a PyTorch implementation of [Listen, Attend and Spell (LAS)](https://arxiv.org/pdf/1508.01211v2.pdf) paper 

```
@article{DBLP:journals/corr/ChanJLV15,
  author    = {William Chan and
               Navdeep Jaitly and
               Quoc V. Le and
               Oriol Vinyals},
  title     = {Listen, Attend and Spell},
  journal   = {CoRR},
  volume    = {abs/1508.01211},
  year      = {2015},
  url       = {http://arxiv.org/abs/1508.01211},
  eprinttype = {arXiv},
  eprint    = {1508.01211},
  timestamp = {Mon, 13 Aug 2018 16:46:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ChanJLV15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# Train on your data
In order to train the model on your data follow the steps below 
### 1. data preprocessing 
* prepare your data and make sure the data is formatted in an CSV format as below 
```
audio_path,text,duration
file/to/file.wav,the text in that file,3.2 
```
* make sure the audios are MONO if not make the proper conversion to meet this condition

### 2. Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```

### 3. Training 
* update the config file if needed
* train the model 
  * from scratch 
  ```bash
  python train.py
  ```
  * from checkpoint 
  ```
  python train.py checkpoint=path/to/checkpoint tokenizer.tokenizer_file=path/to/tokenizer.json
  ```

# TODO
- [ ] Compeleting the inference module 
- [ ] Adding Demo
