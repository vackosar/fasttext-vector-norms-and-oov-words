# FastText Vector Norms And OOV Words


# Abstract

Word embeddings, trained on large unlabeled corpora are useful for many natural language processing tasks. FastText [(Bojanowski et al., 2016)](https://arxiv.org/abs/1607.04606) in contrast to Word2vec model accounts for sub-word information by also embedding sub-word n-grams. FastText word representation is the word embedding vector plus sum of n-grams contained in it.
Word2vec vector norms have been shown [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297) to be correlated to word significance. This blog post visualize vector norms of FastText embedding and evaluates use of FastText word vector norm multiplied with number of word n-grams for detecting non-english OOV words.


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

As mentioned above each FastText vocab word has its vector representation regardless its size. Norms of those vectors are plotted in this section. The shape of the distribution seems to match closely the shape of the same plot for Word2Vec [(Schakel & Wilson, 2015)](http://arxiv.org/abs/1508.02297). The vector norm as measure of word significance seems to hold even for FastText in terms of this norm as can be seen from labeled samples in the scatter plot (same frequency bin with increasing vector norm: authors, Alfine, numbertel).
![no_ngram_norm-tf](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/no_ngram_norm-tf.png)


# NG_Norm (N-Grams Times Count Norm)

As mentioned above FastText uses average of word vectors used. However for detection of noise-words number of ngrams seems to useful. For that purpose NG_Norm is defined <b>ng_norm(word)= || &Sigma;<sub>g &isin; ngrams(word)</sub>v<sub>g</sub> ||</b>. Using this norm common words are clustered in narrower band on ng_norm axis.

![ng_norm-tf](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-tf.png)

Explicitly aggregated distribution on ng_norm axis is plotted in histogram below.
![ng_norm-hist](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-hist.png)

Probability distribution of given FastText vocabulary word being common word is plotted below. The distribution is well approximated by t-distribution.

![ng_norm-common-density](https://raw.githubusercontent.com/vackosar/fasttext-vector-norms-and-oov-words/master/results/ng_norm-common-density.png)


# Norms of Hyponyms vs Hypernyms

To evaluate thesis of (Shakel 2015) that word specificity in given term-frequency norm is correlated with vector norm for FastText 67 pairs of hyponyms and hypernyms are used. From just these few examples we see that No-NGram norm with 77% accuracy predicts which word is hyponym and which hypernym disregarding their term-frequencies. Below is the data used. The norm colums contain relative percent differences i.e. ```(hypo-hyper) / hyper * 100```.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;"> <th></th> <th>hyper</th> <th>hypo</th> <th>standard_norm</th> <th>no_ngram_norm</th> <th>ng_norm</th> <th>count</th> </tr> </thead>
  <tbody>
    <tr> <th>0</th> <td>month</td> <td>January</td> <td>-22.559890</td> <td>12.179197</td> <td>29.066855</td> <td>76.994548</td> </tr>
    <tr> <th>1</th> <td>month</td> <td>February</td> <td>-34.532857</td> <td>13.354693</td> <td>30.934289</td> <td>57.404248</td> </tr>
    <tr> <th>2</th> <td>month</td> <td>March</td> <td>21.790121</td> <td>8.177371</td> <td>21.790129</td> <td>91.525721</td> </tr>
    <tr> <th>3</th> <td>month</td> <td>April</td> <td>25.993046</td> <td>10.371281</td> <td>25.993049</td> <td>86.943093</td> </tr>
    <tr> <th>4</th> <td>month</td> <td>May</td> <td>247.451639</td> <td>6.607942</td> <td>15.817219</td> <td>219.577850</td> </tr>
    <tr> <th>5</th> <td>month</td> <td>June</td> <td>86.636376</td> <td>9.665938</td> <td>24.424255</td> <td>80.363607</td> </tr>
    <tr> <th>6</th> <td>month</td> <td>July</td> <td>93.219042</td> <td>12.777551</td> <td>28.812698</td> <td>71.600872</td> </tr>
    <tr> <th>7</th> <td>month</td> <td>August</td> <td>-4.813989</td> <td>11.601140</td> <td>26.914686</td> <td>56.870358</td> </tr>
    <tr> <th>8</th> <td>month</td> <td>September</td> <td>-44.985682</td> <td>12.394985</td> <td>28.366747</td> <td>61.352990</td> </tr>
    <tr> <th>9</th> <td>month</td> <td>October</td> <td>-21.949212</td> <td>12.073578</td> <td>30.084649</td> <td>64.158556</td> </tr>
    <tr> <th>10</th> <td>month</td> <td>November</td> <td>-35.144106</td> <td>13.222669</td> <td>29.711792</td> <td>55.423498</td> </tr>
    <tr> <th>11</th> <td>month</td> <td>December</td> <td>-34.639645</td> <td>12.905714</td> <td>30.720711</td> <td>59.169547</td> </tr>
    <tr> <th>12</th> <td>color</td> <td>red</td> <td>214.267874</td> <td>-2.442838</td> <td>4.755946</td> <td>-14.315463</td> </tr>
    <tr> <th>13</th> <td>color</td> <td>blue</td> <td>44.778407</td> <td>-5.899290</td> <td>-3.481073</td> <td>-40.683531</td> </tr>
    <tr> <th>14</th> <td>color</td> <td>green</td> <td>-16.087377</td> <td>-4.437130</td> <td>-16.087392</td> <td>-30.029118</td> </tr>
    <tr> <th>15</th> <td>color</td> <td>white</td> <td>-3.950346</td> <td>-4.100787</td> <td>-3.950355</td> <td>24.457167</td> </tr>
    <tr> <th>16</th> <td>color</td> <td>orange</td> <td>-19.920026</td> <td>1.365383</td> <td>6.773289</td> <td>-80.102688</td> </tr>
    <tr> <th>17</th> <td>color</td> <td>purple</td> <td>-25.538990</td> <td>-3.007207</td> <td>-0.718664</td> <td>-87.577665</td> </tr>
    <tr> <th>18</th> <td>color</td> <td>black</td> <td>-5.428290</td> <td>-3.726623</td> <td>-5.428305</td> <td>26.119314</td> </tr>
    <tr> <th>19</th> <td>color</td> <td>pink</td> <td>61.684030</td> <td>1.409762</td> <td>7.789344</td> <td>-74.939234</td> </tr>
    <tr> <th>20</th> <td>color</td> <td>yellow</td> <td>-25.193438</td> <td>1.552819</td> <td>-0.257928</td> <td>-71.494422</td> </tr>
    <tr> <th>21</th> <td>color</td> <td>cyan</td> <td>87.807763</td> <td>18.866982</td> <td>25.205162</td> <td>-99.416056</td> </tr>
    <tr> <th>22</th> <td>color</td> <td>violet</td> <td>-28.965518</td> <td>10.081180</td> <td>-5.287368</td> <td>-98.384650</td> </tr>
    <tr> <th>23</th> <td>color</td> <td>grey</td> <td>39.793408</td> <td>-0.529619</td> <td>-6.804404</td> <td>-86.867216</td> </tr>
    <tr> <th>24</th> <td>animal</td> <td>dog</td> <td>252.456093</td> <td>-6.000990</td> <td>-11.885978</td> <td>76.011544</td> </tr>
    <tr> <th>25</th> <td>animal</td> <td>cat</td> <td>234.540272</td> <td>-6.204253</td> <td>-16.364929</td> <td>-22.229570</td> </tr>
    <tr> <th>26</th> <td>animal</td> <td>bird</td> <td>85.873812</td> <td>2.556551</td> <td>-7.063093</td> <td>-45.238106</td> </tr>
    <tr> <th>27</th> <td>animal</td> <td>reptile</td> <td>-22.501929</td> <td>15.213460</td> <td>-3.127411</td> <td>-97.982462</td> </tr>
    <tr> <th>28</th> <td>animal</td> <td>fish</td> <td>75.872201</td> <td>2.111702</td> <td>-12.063900</td> <td>22.179526</td> </tr>
    <tr> <th>29</th> <td>animal</td> <td>cow</td> <td>267.844129</td> <td>7.508819</td> <td>-8.038966</td> <td>-82.955819</td> </tr>
    <tr> <th>30</th> <td>animal</td> <td>insect</td> <td>-7.264797</td> <td>7.887063</td> <td>-7.264797</td> <td>-90.300180</td> </tr>
    <tr> <th>31</th> <td>animal</td> <td>fly</td> <td>259.201598</td> <td>-6.115292</td> <td>-10.199600</td> <td>-24.141623</td> </tr>
    <tr> <th>32</th> <td>animal</td> <td>mammal</td> <td>3.345599</td> <td>16.280858</td> <td>3.345599</td> <td>-96.894255</td> </tr>
    <tr> <th>33</th> <td>tool</td> <td>hammer</td> <td>-60.361552</td> <td>5.061610</td> <td>-20.723104</td> <td>-88.923340</td> </tr>
    <tr> <th>34</th> <td>tool</td> <td>screwdriver</td> <td>-75.439298</td> <td>33.551341</td> <td>10.523150</td> <td>-97.639422</td> </tr>
    <tr> <th>35</th> <td>tool</td> <td>drill</td> <td>-43.531817</td> <td>11.923173</td> <td>-15.297724</td> <td>-85.132555</td> </tr>
    <tr> <th>36</th> <td>tool</td> <td>handsaw</td> <td>-49.962413</td> <td>76.553452</td> <td>25.093964</td> <td>-99.873156</td> </tr>
    <tr> <th>37</th> <td>tool</td> <td>knife</td> <td>-37.435886</td> <td>20.666681</td> <td>-6.153829</td> <td>-75.100349</td> </tr>
    <tr> <th>38</th> <td>tool</td> <td>wrench</td> <td>-51.094025</td> <td>26.541042</td> <td>-2.188051</td> <td>-96.230430</td> </tr>
    <tr> <th>39</th> <td>tool</td> <td>pliers</td> <td>-45.382544</td> <td>50.950378</td> <td>9.234910</td> <td>-98.404390</td> </tr>
    <tr> <th>40</th> <td>fruit</td> <td>banana</td> <td>-22.010443</td> <td>-1.040847</td> <td>3.986077</td> <td>-81.245291</td> </tr>
    <tr> <th>41</th> <td>fruit</td> <td>apple</td> <td>-3.168075</td> <td>-1.298223</td> <td>-3.168080</td> <td>-56.612444</td> </tr>
    <tr> <th>42</th> <td>fruit</td> <td>pear</td> <td>59.400398</td> <td>5.798260</td> <td>6.266932</td> <td>-93.565996</td> </tr>
    <tr> <th>43</th> <td>fruit</td> <td>peach</td> <td>-3.104994</td> <td>-7.888756</td> <td>-3.105001</td> <td>-91.252127</td> </tr>
    <tr> <th>44</th> <td>fruit</td> <td>orange</td> <td>-27.728805</td> <td>-11.572789</td> <td>-3.638405</td> <td>-36.933547</td> </tr>
    <tr> <th>45</th> <td>fruit</td> <td>pineapple</td> <td>-55.388695</td> <td>2.253465</td> <td>4.093046</td> <td>-91.038789</td> </tr>
    <tr> <th>46</th> <td>fruit</td> <td>lemon</td> <td>7.380923</td> <td>0.148509</td> <td>7.380918</td> <td>-65.358937</td> </tr>
    <tr> <th>47</th> <td>fruit</td> <td>pomegranate</td> <td>-63.004678</td> <td>3.623020</td> <td>10.985970</td> <td>-97.139377</td> </tr>
    <tr> <th>48</th> <td>fruit</td> <td>grape</td> <td>5.267917</td> <td>6.485440</td> <td>5.267921</td> <td>-88.986123</td> </tr>
    <tr> <th>49</th> <td>fruit</td> <td>strawberries</td> <td>-65.290713</td> <td>4.979606</td> <td>15.697627</td> <td>-88.793171</td> </tr>
    <tr> <th>50</th> <td>flower</td> <td>peony</td> <td>51.811230</td> <td>15.825447</td> <td>13.858414</td> <td>-98.449092</td> </tr>
    <tr> <th>51</th> <td>flower</td> <td>rose</td> <td>146.126568</td> <td>-2.189412</td> <td>23.063286</td> <td>16.749135</td> </tr>
    <tr> <th>52</th> <td>flower</td> <td>lily</td> <td>108.582103</td> <td>7.221601</td> <td>4.291051</td> <td>-93.181922</td> </tr>
    <tr> <th>53</th> <td>flower</td> <td>tulip</td> <td>49.614036</td> <td>14.132214</td> <td>12.210532</td> <td>-96.028388</td> </tr>
    <tr> <th>54</th> <td>flower</td> <td>sunflower</td> <td>-34.754577</td> <td>9.156723</td> <td>14.179479</td> <td>-90.751123</td> </tr>
    <tr> <th>55</th> <td>flower</td> <td>marigold</td> <td>-25.294849</td> <td>9.122744</td> <td>12.057728</td> <td>-99.084527</td> </tr>
    <tr> <th>56</th> <td>flower</td> <td>orchid</td> <td>10.703952</td> <td>7.983661</td> <td>10.703952</td> <td>-93.583253</td> </tr>
    <tr> <th>57</th> <td>tree</td> <td>pine</td> <td>9.169017</td> <td>4.525833</td> <td>9.169017</td> <td>-86.723162</td> </tr>
    <tr> <th>58</th> <td>tree</td> <td>pear</td> <td>20.911832</td> <td>11.848664</td> <td>20.911832</td> <td>-95.662470</td> </tr>
    <tr> <th>59</th> <td>tree</td> <td>maple</td> <td>-12.181924</td> <td>19.677509</td> <td>31.727102</td> <td>-90.921096</td> </tr>
    <tr> <th>60</th> <td>tree</td> <td>oak</td> <td>155.620050</td> <td>15.579368</td> <td>27.810028</td> <td>-85.239221</td> </tr>
    <tr> <th>61</th> <td>tree</td> <td>aspen</td> <td>-16.014889</td> <td>16.366398</td> <td>25.977674</td> <td>-99.318727</td> </tr>
    <tr> <th>62</th> <td>tree</td> <td>spruce</td> <td>-32.479379</td> <td>5.101566</td> <td>35.041240</td> <td>-97.131057</td> </tr>
    <tr> <th>63</th> <td>tree</td> <td>larch</td> <td>-4.069312</td> <td>22.811006</td> <td>43.896031</td> <td>-99.657494</td> </tr>
    <tr> <th>64</th> <td>tree</td> <td>linden</td> <td>-46.235502</td> <td>23.560859</td> <td>7.528996</td> <td>-99.731559</td> </tr>
    <tr> <th>65</th> <td>tree</td> <td>juniper</td> <td>-57.948077</td> <td>14.995041</td> <td>5.129804</td> <td>-99.002917</td> </tr>
    <tr> <th>66</th> <td>tree</td> <td>birch</td> <td>-20.747948</td> <td>14.309570</td> <td>18.878067</td> <td>-97.591876</td> </tr>
    <tr> <th>67</th> <td>tree</td> <td>elm</td> <td>196.460629</td> <td>20.488973</td> <td>48.230311</td> <td>-98.977328</td> </tr>
    <tr> <th>68</th> <td>average</td> <td></td> <td>25.257317</td> <td>9.337584</td> <td>9.726517</td> <td>-44.851693</td> </tr>
    <tr> <th>69</th> <td>counts</td> <td></td> <td>42.647059</td> <td>77.941176</td> <td>66.176471</td> <td>NaN</td> </tr>
    <tr> <th>70</th> <td>counts selected</td> <td></td> <td>42.647059</td> <td>77.941176</td> <td>66.176471</td> <td>NaN</td> </tr>
  </tbody>
</table>


# Detecting non-english words using NG_Norm

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

- [See my blog for more articles like this or mail me at admin@vaclavkosar.com](https://vaclavkosar.com)


# References

- [Piotr  Bojanowski,   Edouard  Grave,   Armand  Joulin,and  Tomas  Mikolov.  2016.    Enriching  word  vec-tors  with  subword  information. arXiv preprint arXiv:1607.04606.](https://arxiv.org/abs/1607.04606)
- [Adriaan M. J. Schakel and Benjamin J Wilson.  Measuring Word Significance using Distributed Representations of Words. aug 2015. http://arxiv.org/abs/1508.02297](http://arxiv.org/abs/1508.02297).
- [Vitalii  Zhelezniak,  Aleksandar  Savkov,  April  Shen,Francesco  Moramarco,   Jack  Flann,   and  Nils  Y.Hammerla. 2019.   Don’t settle for average,  go for the max:  Fuzzy sets and max-pooled word vectors. In International Conference on Learning Representations.](https://arxiv.org/abs/1904.13264)