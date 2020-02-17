# nnvlp - A Neural Network-Based Vietnamese Language Processing Toolkit
-----------------------------------------------------------------
Code by **Thai-Hoang Pham** at Alt Inc. (Utilize some code at a [repository](https://github.com/XuezheMax/LasagneNLP))

A demo website is available at [nnvlp.org](http://nnvlp.org)

## 1. Introduction
**nnvlp** is a Python package of the system described in a paper [NNVLP: A Neural Network-Based Vietnamese 
Language Processing Toolkit](https://arxiv.org/abs/1708.07241).
This package is used for some common sequence labeling tasks for Vietnamese including part-of-speech (POS) tagging, 
chunking, named entity recognition (NER).
The architecture of a model in this package is the combination of bi-directional Long Short-Term Memory (Bi-LSTM), 
Conditional Random Field (CRF), and word embeddings that is the concatenation of pre-trained word embeddings learnt 
from skip-gram model and character-level word features learnt from Convolutional Neural Network (CNN).

Our package achieves an accuracy of 91.92%, F1 scores of 84.11% and 92.91% for POS tagging, chunking, and NER tasks 
respectively.

The following tables compare the performance of **nnvlp** and other previous toolkit on POS tagging, chunking, and NER 
task respectively.

### POS tagging

| System       | Accuracy |
|--------------|----------|
| Vitk         | 88.41    |
| vTools       | 90.73    |
| RDRPOSTagger | 91.96    |
| nnvlp        | **91.92**    |

### Chunking

| System | P     | R     | F1    |
|--------|-------|-------|-------|
| vTools | 82.79 | 83.55 | 83.17 |
| nnvlp  | 83.93 | 84.28 | **84.11** |

### NER

| System       | P     | R     | F1    |
|--------------|-------|-------|-------|
| Vitk         | 88.36 | 89.20 | 88.78 |
| vie-ner-lstm | 91.09 | 93.03 | 92.05 |
| nnvlp        | 92.76 | 93.07 | **92.91** |

## 2. Installation

The simple way to install **nnvlp** is using pip:

```sh
    $ pip install nnvlp
```
## 3. Usage

```sh
    >>> import nnvlp
    >>> model = nnvlp.NNVLP()
    >>> output = model.predict(u"Hôm nay tôi ra Hà Nội gặp ông Nam. Ông Nam là giảng viên đại học Bách Khoa.")
```

The default output is a dict that contains **token_text**, **pos**, **chunk**, **ner** attributes. At this version, 
there are two other display formats including **CoNLL** and **JSON**. If you want to get easy-to-read format, you can use 
**CoNLL** option.

```sh
    >>> import nnvlp
    >>> model = nnvlp.NNVLP()
    >>> output = model.predict(u"Hôm nay tôi ra Hà Nội gặp ông Nam. Ông Nam là giảng viên đại học Bách Khoa.", display_format="CoNLL")
    >>> print(output)
    1	Hôm_nay		N		B-NP		O
    2	tôi		P		B-NP		O
    3	ra		V		B-VP		O
    4	Hà_Nội		Np		B-NP		B-LOC
    5	gặp		V		B-VP		O
    6	ông		Nc		B-NP		O
    7	Nam		Np		I-NP		B-PER
    8	.		CH		O		O
    
    1	Ông		Nc		B-NP		O
    2	Nam		Np		I-NP		B-PER
    3	là		V		B-VP		O
    4	giảng_viên		N		B-NP		O
    5	đại_học		N		B-NP		B-ORG
    6	Bách_Khoa		Np		I-NP		I-ORG
    7	.		CH		O		O
```

**Note**: This version works only for **MAC OS** and **Linux** environments. Installing **nnvlp** package may take a few minutes 
because it downloads Vietnamese word embeddings and **NLTK** data from Internet if you don't have them installed before.

## 4. References

[Thai-Hoang Pham, Xuan-Khoai Pham, Tuan-Anh Nguyen, Phuong Le-Hong, "NNVLP: A Neural Network-Based Vietnamese Language 
Processing Toolkit" Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)](https://arxiv.org/abs/1708.07241)

```
@inproceedings{Pham:2017b,
  title={NNVLP: A Neural Network-Based Vietnamese Language Processing Toolkit},
  author={Thai-Hoang Pham and Xuan-Khoai Pham and Tuan-Anh Nguyen and Phuong Le-Hong},
  booktitle={Proceedings of The 8th International Joint Conference on Natural Language Processing},
  year={2017},
}
```

[Thai-Hoang Pham, Phuong Le-Hong, "End-to-end Recurrent Neural Network Models for Vietnamese Named Entity Recognition: 
Word-level vs. Character-level" Proceedings of The 15th International Conference of the Pacific Association for 
Computational Linguistics (PACLING 2017)](https://arxiv.org/abs/1705.04044)

```
@inproceedings{Pham:2017a,
  title={End-to-end Recurrent Neural Network Models for Vietnamese Named Entity Recognition: Word-level vs. Character-level},
  author={Thai-Hoang Pham and Phuong Le-Hong},
  booktitle={Proceedings of The 15th International Conference of the Pacific Association for Computational Linguistics},
  year={2017},
}
```

## 5. Contact

**Thai-Hoang Pham** < phamthaihoang.hn@gmail.com >

Alt Inc, Hanoi, Vietnam
