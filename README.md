# FastText Vector Norms And OOV Words

# Abstract

Word embeddings, trained on large unlabeled corpora are useful for many natural language processing tasks. FastText model introduced in [(Bojanowski et al., 2016)](https://arxiv.org/abs/1607.04606) in contrast to Word2vec model accounts for sub-word information by embeddings also word n-grams. FastText word representation is whole word vector plus sum of n-grams contained in it.
Word2vec vector norms have been show in [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297) to be correlated to word significance. This blog post visualize vector norms of FastText embedding and evaluates use of FastText word vector norm multiplied with number of word n-grams for detecting non-english OOV words.

# Introduction

FastText embeds words by adding word's n-grams to the words embedding and then normalizes by total token count i.e. <b>fastText(word)<sub></sub> = (v<sub>word</sub> + &Sigma;<sub>g &in; ngrams(word)</sub>v<sub>g</sub>) / (1 + |ngrams(word)|)</b>. However if the word if not present in the dictionary (OOV) only n-grams are used i.e. <b>fastText(word) = (&Sigma;<sub>g &in; ngrams(word)</sub>v<sub>g</sub>) / |ngrams(word)|</b>. For purpose of studying OOV words the asymmetry between vocabulary and out of vocabulary words is removed by only utilizing word's n-grams regardless if it is OOV or not.

In order to study contrast between common english words e.g. "apple" and noise words (usually some parsing artifacts or unusual tokens with very specific meaning) e.g. "wales-2708" or "G705" [MIT 10K Common words dataset is used](https://www.mit.edu/~ecprice/wordlist.10000).

Entire code for this post in available in [this repository in file "main.py"](https://github.com/vackosar/fasttext-vector-norms-and-oov-words/blob/master/main.py). FastText model used is [5-gram English 2M "cc.en.300.bin"](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz).

# Standard Vector Norm

Standard vector norm as defined in Gensim implementation is used in this section.

![]()


# Conclusion



# References

- [Piotr  Bojanowski,   Edouard  Grave,   Armand  Joulin,and  Tomas  Mikolov.  2016.    Enriching  word  vec-tors  with  subword  information. arXiv preprint arXiv:1607.04606.](https://arxiv.org/abs/1607.04606)
- [Adriaan M. J. Schakel and Benjamin J Wilson.  Measuring Word Significance using DistributedRepresentations of Words. aug 2015. http://arxiv.org/abs/1508.02297](http://arxiv.org/abs/1508.02297).