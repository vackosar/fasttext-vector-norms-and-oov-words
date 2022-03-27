# FastText Vector Norms And OOV Words


# Summary

Word embeddings, trained on large unlabeled corpora are useful for many natural language processing tasks. FastText [(Bojanowski et al., 2016)](https://arxiv.org/abs/1607.04606) in contrast to Word2vec model accounts for sub-word information by also embedding sub-word n-grams. FastText word representation is the word embedding vector plus sum of n-grams contained in it.
Word2vec vector norms have been shown [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297) to be correlated to word significance. This blog post visualize vector norms of FastText embedding and evaluates use of FastText word vector norm multiplied with number of word n-grams for detecting non-english OOV words.

- [Read full description of this experiment on Fasttext OOV on my blog and ask or subscribe](https://vaclavkosar.com/ml/FastText-Vector-Norms-And-OOV-Words)
- [Entire code for this post in available in this repository in file "main.py"](https://github.com/vackosar/fasttext-vector-norms-and-oov-words/blob/master/main.py)
- [Continue: StarSpace - a general-purpose embeddings inspired by FastText](https://vaclavkosar.com/ml/starspace-embedding)
