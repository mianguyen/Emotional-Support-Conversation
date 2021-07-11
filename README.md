# Emotional-Support-Conversation
Data and codes for the ACL 2021 paper: [**Towards Emotional Support Dialog Systems**](https://arxiv.org/abs/2106.01144)

If you use our codes or your research is related to our paper, please kindly cite our paper:

```bib
@inproceedings{liu-etal-2021-towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang  and 
    Zheng, Chujie  and 
    Demasi, Orianna  and 
    Sabour, Sahand  and 
    Li, Yu  and 
    Yu, Zhou  and 
    Jiang, Yong  and 
    Huang, Minlie},
  booktitle={Proceedings of the 59th annual meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Data

The corpus file is `ESConv.json`. We have collected **more** conversations with more problem topics. ESConv now contians 1,410 conversations with 10 topic problems.

### Statistics
#### Problem Category

| Problem Category | ongoing depression | breakup with partner | job crisis | problems with friends | academic pressure | procras-<br>tination* | alcohol abuse* | issues with parent* | sleep problems* |  appearance anxiety* | school bullying* | issues with children* |
| :-------- | :---------- | :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- | :---------- | :---------- | 
| Number| 386 |260 | 317 | 191 | 172 |  13 | 12 | 18 | 28 | 12 | 2 | 10 |


\* denotes the new topics that we add for the second collection. The new data is aimed to support the research in transferring the model's ability from old topics to the new ones. 

<font size=1>
  
#### Strategy Category 
| Strategy Category| Number   |
| :--------------  | :------- |
| Questions | 4138(20.7%)|
| Self-disclosure | 1836(9.1%) |
| Affirmation and Reassurance | 3058(15.3%) |
| Providing Suggestions | 3237(16.2%) |
| Other | 3661(18.3%) |
| Reflection of feelings | 1580(7.9%) | 
| Information | 1316(6.6%) | 
| Restatement or Paraphrasing | 1184(5.9%) |
  
</font>


## Model Implementation

We provide two versions of model implementation:

- `codes` is the version that we used in the original experiments
- `codes_zcj` is the version reproduced by  [@chujiezheng](https://github.com/chujiezheng)



