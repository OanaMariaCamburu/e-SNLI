# News

**`!`** Check out our e-SNLI-VE, a new dataset of natural language explanations for vision-language understanding, and our e-ViL benchmark for evaluating natural language explanations: [e-ViL: A Dataset and Benchmark for Natural Language Explanations in
Vision-Language Tasks](https://openaccess.thecvf.com/content/ICCV2021/papers/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.pdf) accepted at ICCV, 2021.

**`!`** New work on e-SNLI: [Make Up Your Mind! Adversarial Generation of Inconsistent Natural Language Explanations](https://arxiv.org/abs/1910.03065). Accepted as a short paper at ACL, 2020.

**`!`** New dataset of visual textual entailment with natural language explanations taken from e-SNLI: [e-SNLI-VE-2.0: Corrected Visual-Textual Entailment with Natural Language Explanations](https://arxiv.org/abs/2004.03744). At the IEEE CVPR Workshop on Fair, Data Efficient and Trusted Computer Vision, 2020

**`!`** If are also interested in feature-based explanations besides natural language explanations, check out our new works on:

* verifying post-hoc explanatory methods: [Can I Trust the Explainer? Verifying Post-hoc Explanatory Methods](https://arxiv.org/abs/1910.02065). At NeurIPS 2019 Workshop on Safety and Robustness in Decision Making, 2020.
* the problems encournteres while trying to explain using only input features: [The Struggles of Feature-Based Explanations: Shapley Values vs. Minimal Sufficient Subsets](https://arxiv.org/abs/2009.11023).

# e-SNLI
There are 2 splits for the train set due to the github sie restrictions, please simply merge them.

Clarification on the two potentially confusing headers:

* Sentence1_marked_1: is the premise (Sentence2 for hypothesis) were words between star (*) were highlighted by the annotators. The annotators had to click on every word individually to highlight it. The punctuation has not been separated from the words, hence highlighting a word automatically included any punctuation near it.
Please use only this header to retrieve the highlighted words simply by retrieving the words between stars without space between them, i.e., things like *w1* w2 *w3* only w1 and w3 were highlighted. 

Please *ignore* the fields Sentence_Highlighted_ and retrieve the highlighted words from the Sentence_marked_ fields as stated above.



# Trained models
Trained models can be downloaded at:
* PredictAndExplain: https://drive.google.com/file/d/1w8UlNQ5yvZPNu4RgVkgICB6qsefkjolG/view?usp=sharing
* ExplainThenPredictAttention: https://drive.google.com/file/d/1l7dnml7mDnT72QrwZMmA7VGIsWjVpQT6/view?usp=sharing
* ExplanationsToLabels: https://drive.google.com/file/d/1_rFGlFYHSJ1xqjA2lDjzBvO5mf7INo1A/view?usp=sharing

# Dependancies
* Python 2.7
* Pytorch 0.3.1
* NLTK >= 3

# Bibtex
If you use this dataset or code in your work, please cite [our paper](https://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf):
```
@incollection{NIPS2018_8163,
title = {e-SNLI: Natural Language Inference with Natural Language Explanations},
author = {Camburu, Oana-Maria and Rockt\"{a}schel, Tim and Lukasiewicz, Thomas and Blunsom, Phil},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {9539--9549},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf}
}
