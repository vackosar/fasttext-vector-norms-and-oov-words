# FastText Vector Norms And OOV Words

# Abstract

Word embeddings, trained on large unlabeled corpora are useful for many natural language processing tasks. FastText [(Bojanowski et al., 2016)](https://arxiv.org/abs/1607.04606) in contrast to Word2vec model accounts for sub-word information by also embedding sub-word n-grams. FastText word representation is the word embedding vector plus sum of n-grams contained in it.
Word2vec vector norms have been show [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297) to be correlated to word significance. This blog post visualize vector norms of FastText embedding and evaluates use of FastText word vector norm multiplied with number of word n-grams for detecting non-english OOV words.

# Introduction

FastText embeds words by adding word's n-grams to the word embedding and then normalizes by total token count i.e. <b>fastText(word)<sub></sub> = (v<sub>word</sub> + &Sigma;<sub>g &isin; ngrams(word)</sub>v<sub>g</sub>) / (1 + |ngrams(word)|)</b>. However if the word is not present in the dictionary (OOV) only n-grams are used i.e. <b>(&Sigma;<sub>g &isin; ngrams(word)</sub>v<sub>g</sub>) / |ngrams(word)|</b>. For purpose of studying OOV words this asymmetry between vocabulary and out of vocabulary words is removed by only utilizing word's n-grams regardless if the word is OOV or not.

In order to study contrast between common english words e.g. "apple" and noise-words (usually some parsing artifacts or unusual tokens with very specific meaning) e.g. "wales-2708" or "G705" [MIT 10K Common words dataset is used](https://www.mit.edu/~ecprice/wordlist.10000).

Entire code for this post in available in [this repository in file "main.py"](https://github.com/vackosar/fasttext-vector-norms-and-oov-words/blob/master/main.py). FastText model used is [5-gram English 2M "cc.en.300.bin"](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz).

# Standard Vector Norm

Standard vector norm as defined in Gensim implementation is used in this section. Common words are located mostly on the right in the term-frequency spectrum and clustered in three different areas in the norm spectrum. On both axis common words are clustered approximatelly in 4 areas.


![standard_norm-tf](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/standard_norm-tf.png)

From below samples it is not clear what clusters correspond to:

- bottom left cluster: now, three, month, News, Big, picked, votes, signature, Challenge, Short, trick, Lots, 68, priorities, upgrades
- bottom right cluster: our, home, game, won, control, law, common, Street, speed, Tuesday, direct, helped, passed, condition, Date, signed
- middle right cluster: via, companies, necessary, straight, menu, kinds, Championship, relief, periods, Prize, minimal, Rated, 83, wears
- top right cluster: position, wonderful, shooting, switch, â, Atlantic, ladies, vegetables, tourist, HERE, prescription, upgraded, Evil

# No N-Gram Norm

As mentioned above each FastText vocab word has its vector representation regardless its size. Norms of those vectors are plotted in this section. The shape of the distribution seems to match closely the shape of the same plot for Word2Vec [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297). The vector norm as measure of word significance seems to hold even for FastText in terms of this norm as can be seen from labeled samples in the scatter plot.
![no_ngram_norm-tf](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/no_ngram_norm-tf.png)


# NG_Norm (N-Grams Times Count Norm)

As mentioned above FastText uses average of word vectors used. However for detection of noise-words number of ngrams seems to useful. For that purpose NG_Norm is defined <b>ng_norm(word)= || &Sigma;<sub>g &isin; ngrams(word)</sub>v<sub>g</sub> ||</b>. Using this norm common words are clustered in narrower band on ng_norm axis.

![ng_norm-tf](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-tf.png)

Explicitly aggregated distribution on ng_norm axis is plotted in histogram below.
![ng_norm-hist](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-hist.png)

Probability distribution of given FastText vocabulary word being common word is plotted below. The distribution is well approximated by t-distribution.

![ng_norm-common-density](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-common-density.png)

Ability to detect noisy-words is evaluated on simple task of splitting two concatenated words back apart below. For example let's split back concatenation 'inflationlithium':

<table border="1" class="dataframe">
  <thead><tr style="text-align: right;"> <th>word1</th> <th>word2</th> <th>norm1</th> <th>norm2</th> <th>prob1</th> <th>prob2</th> <th>prob</th> </tr> </thead>
  <tbody>
    <tr> <td>i</td> <td>nflationlithium</td> <td>0</td> <td>4.20137</td> <td>0.000000</td> <td>0.000397</td> <td>0.000000e+00</td> </tr>
    <tr> <td>in</td> <td>flationlithium</td> <td>0</td> <td>4.40944</td> <td>0.000000</td> <td>0.000519</td> <td>0.000000e+00</td> </tr>
    <tr> <td>inf</td> <td>lationlithium</td> <td>1.88772</td> <td>3.86235</td> <td>0.010414</td> <td>0.000741</td> <td>7.721472e-06</td> </tr> 
    <tr> <td>infl</td> <td>ationlithium</td> <td>2.29234</td> <td>4.04391</td> <td>0.053977</td> <td>0.000428</td> <td>2.308942e-05</td> </tr>
    <tr> <td>infla</td> <td>tionlithium</td> <td>2.24394</td> <td>4.74456</td> <td>0.052467</td> <td>0.000000</td> <td>0.000000e+00</td> </tr>
    <tr> <td>inflat</td> <td>ionlithium</td> <td>2.55929</td> <td>3.45802</td> <td>0.048715</td> <td>0.002442</td> <td>1.189513e-04</td> </tr>
    <tr> <td>inflati</td> <td>onlithium</td> <td>3.10228</td> <td>3.55187</td> <td>0.007973</td> <td>0.001767</td> <td>1.408828e-05</td> </tr>
    <tr> <td>inflatio</td> <td>nlithium</td> <td>3.34667</td> <td>3.26616</td> <td>0.003907</td> <td>0.003159</td> <td>1.234263e-05</td> </tr>
    <tr> <td>inflation</td> <td>lithium</td> <td>2.87083</td> <td>2.73886</td> <td>0.017853</td> <td>0.035389</td> <td>6.318213e-04</td> </tr>
    <tr> <td>inflationl</td> <td>ithium</td> <td>3.36933</td> <td>2.35156</td> <td>0.002887</td> <td>0.053333</td> <td>1.539945e-04</td> </tr>
    <tr> <td>inflationli</td> <td>thium</td> <td>3.73344</td> <td>2.21766</td> <td>0.001283</td> <td>0.052467</td> <td>6.730259e-05</td> </tr>
    <tr> <td>inflationlit</td> <td>hium</td> <td>4.16165</td> <td>1.66477</td> <td>0.000096</td> <td>0.004324</td> <td>4.139165e-07</td> </tr>
    <tr> <td>inflationlith</td> <td>ium</td> <td>4.40217</td> <td>1.59184</td> <td>0.000519</td> <td>0.002212</td> <td>1.147982e-06</td> </tr>
    <tr> <td>inflationlithi</td> <td>um</td> <td>4.71089</td> <td>0</td> <td>0.000000</td> <td>0.000000</td> <td>0.000000e+00</td> </tr>
    <tr> <td>inflationlithiu</td> <td>m</td> <td>4.91263</td> <td>0</td> <td>0.000213</td> <td>0.000000</td> <td>0.000000e+00</td> </tr>
  </tbody>
</table>
 
Above approach yielded around 48% accuracy on 3000 random two-word samples from MIT 10k common words. A more efficient method in this specific case would be to search vocabulary instead of calculating vector norms. More appropriate comparison however would be for more general task involving OOV words e.g. using Edit Distance performed also on OOV words and words with typos.
 

# Conclusion

FastText vector norms and their term-frequency were visualized and investigated in this post.

Standard Norm Term-Frequency plot revealed potentially interesting clustering of common vectors in three to four main areas.

No-N-Gram Norm has very similar Norm-TF distribution as Word2Vec shown in [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297). The word significance correlation does seem to hold even for FastText embeddings in terms of No-N-Gram Norm.

NG_Norm shows that n-gram count could be potentially useful feature and that simple averaging over n-gram vectors may not be optimal. Perhaps some approach akin to [(Zhelezniak et al., 2019)](https://arxiv.org/abs/1904.13264) could be used.


# References

- [Piotr  Bojanowski,   Edouard  Grave,   Armand  Joulin,and  Tomas  Mikolov.  2016.    Enriching  word  vec-tors  with  subword  information. arXiv preprint arXiv:1607.04606.](https://arxiv.org/abs/1607.04606)
- [Adriaan M. J. Schakel and Benjamin J Wilson.  Measuring Word Significance using DistributedRepresentations of Words. aug 2015. http://arxiv.org/abs/1508.02297](http://arxiv.org/abs/1508.02297).
- [Vitalii  Zhelezniak,  Aleksandar  Savkov,  April  Shen,Francesco  Moramarco,   Jack  Flann,   and  Nils  Y.Hammerla. 2019.   Don’t settle for average,  go for the max:  Fuzzy sets and max-pooled word vectors. In International Conference on Learning Representations.](https://arxiv.org/abs/1904.13264)