from math import sqrt
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from gensim import matutils
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.utils_any2vec import ft_ngram_hashes
from gensim.test.utils import datapath
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import linalg as LA
from numpy.core.multiarray import ndarray
from scipy.optimize import leastsq
from scipy.stats import t

#%% load model from disk

# FIXME use working dir
cap_path = datapath("/home/vackosar/src/fasttext-vector-norms-and-oov-words/data/input/cc.en.300.bin")
# fb_model = load_facebook_model(cap_path)
wv = load_facebook_vectors(cap_path)
wv.init_sims()
print(f'model: maxn: {wv.max_n}, minn {wv.min_n}, vocab size: {len(wv.vectors_vocab)}')


#%% shared methods
tf_label = 'tf (Fasttext Model Word Count)'
ng_norm_label = 'ng_norm = ngram_count * ngram_only_norm i.e. (only sub-ngrams used)'
mit_10k_common_label = 'MIT 10k Common Words'
fasttext_model_vocab_label = 'Fasttext Model 2M Vocabulary Words'


def select_word_index(min_count: int, max_count: int, min_norm: float, max_norm: float) -> int:
    vectors_vocab = wv.vectors_vocab
    for i, v in enumerate(vectors_vocab):
        word = wv.index2word[i]
        # norms[i] = LA.norm(wv.word_vec(word))
        norm = LA.norm(custom_vec(word))
        tf = wv.vocab[word].count
        if max_norm > norm > min_norm and max_count > tf > min_count:
            return i


def common_words_norms(get_vec: Callable):
    words: ndarray = pd.read_csv('data/input/mit-10k-words.csv', header=None, names=['word'])['word'].values
    norms = []
    tfs = []
    for i, word in enumerate(words):
        if word in wv.vocab:
            vocab_word_ = wv.vocab[word]
            norms.append(LA.norm(get_vec(word)))
            # norms.append(LA.norm(wv.vectors_vocab[vocab_word_.index]))
            tfs.append(vocab_word_.count)

    norms = np.array(norms)
    tfs = np.array(tfs)
    non_zero_norms_mask = (norms != 0) & (tfs != 0)
    norms = norms[non_zero_norms_mask]
    tfs = tfs[non_zero_norms_mask]

    print(f'norms {norms.shape}, tfs {tfs.shape}')
    return norms, tfs


def custom_vec(w: str):
    word_vec = np.zeros(wv.vectors_ngrams.shape[1], dtype=np.float32)
    ngram_hashes = ft_ngram_hashes(w, wv.min_n, wv.max_n, wv.bucket, wv.compatible_hash)
    for nh in ngram_hashes:
        word_vec += wv.vectors_ngrams[nh]
    # +1 same as in the adjust vecs method
    #word_vec /= len(ngram_hashes)
    # word_vec /= math.log(1 + len(ngram_hashes))
    return word_vec


def standard_vec(w: str):
    word_vec = np.zeros(wv.vectors_ngrams.shape[1], dtype=np.float32)
    ngram_hashes = ft_ngram_hashes(w, wv.min_n, wv.max_n, wv.bucket, wv.compatible_hash)
    for nh in ngram_hashes:
        word_vec += wv.vectors_ngrams[nh]
    # +1 same as in the adjust vecs method
    if len(ngram_hashes) == 0:
        word_vec.fill(0)
        return word_vec

    else:
        return word_vec / len(ngram_hashes)


def calc_norms(get_vec: Callable):
    norms = np.zeros(len(wv.vectors_vocab), dtype=np.float64)
    tfs = np.zeros(len(wv.vectors_vocab), dtype=np.float64)
    # for i in range(len(vectors_vocab)):
    for word, val in wv.vocab.items():
        # norms[i] = LA.norm(v)
        # word = wv.index2word[i]
        # norms[i] = LA.norm(wv.word_vec(word))
        i = val.index
        norms[i] = LA.norm(get_vec(word))
        # tfs[i] = log(wv.vocab[word].count)
        tfs[i] = val.count

    non_zero_norms_mask = (norms != 0) & (tfs != 0)
    norms = norms[non_zero_norms_mask]
    tfs = tfs[non_zero_norms_mask]

    return norms, tfs


def common_word_norm_density_histogram() -> (ndarray, ndarray):
    common_norms, _ = common_words_norms(custom_vec)
    norms, _ = calc_norms(custom_vec)
    bins = np.linspace(0, 10, 300)
    norm_histogram, _ = np.histogram(norms, bins)
    norm_histogram[0] = 1
    common_histogram, _ = np.histogram(common_norms, bins)
    histogram = common_histogram / norm_histogram
    common_non_zero_on_nan = np.argwhere(common_histogram[np.isnan(histogram)] != 0)
    if len(np.argwhere(common_non_zero_on_nan)) > 0:
        raise ValueError(f'unexpected nan at {common_non_zero_on_nan}, common: {common_histogram[common_non_zero_on_nan]}')

    histogram[np.isnan(histogram)] = 0
    density_histogram = histogram / histogram[np.isfinite(histogram)].sum() / (bins[1] - bins[0])
    return density_histogram, bins


def histogram_position(bins, value) -> int:
    for i, b in enumerate(bins):
        if value < b:
            if i == 0:
                raise ValueError
            else:
                return i - 1
    raise ValueError


def histogram_val(histogram: ndarray, bins: ndarray, value: float) -> float:
    return histogram[np.digitize(value, bins)]


def no_ngram_vector(word: str) -> ndarray:
    if word in wv.vocab:
        return wv.vectors_vocab[wv.vocab[word].index]

    else:
        return np.zeros(wv.vectors_vocab[0].shape[0])


#%% def plot_custom_vec_norms():
norms, tfs = calc_norms(custom_vec)

seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText NG_Norm - TF')
plt.xlabel(tf_label)
plt.xscale('log')
plt.ylabel(ng_norm_label)
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label=fasttext_model_vocab_label)

common_words_norm, common_words_tfs = common_words_norms(custom_vec)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label=mit_10k_common_label)

sorted_idxs = matutils.argsort(norms, reverse=True)

rnd_word_idx = [
    sorted_idxs[400000], sorted_idxs[800000], sorted_idxs[1200000], sorted_idxs[1600000], sorted_idxs[1800000]
    # select_word_index(70_000, 100_000, 2.4, 2.71),
    # select_word_index(55_000, 70_000, 0.53, 0.6),
    # select_word_index(4600_000, 7000_000, 0.44, 0.47),
    # select_word_index(4600_000, 7000_000, 1.26, 1.3),
    # select_word_index(4600_000, 7000_000, 2.4, 2.71)
]
for i in rnd_word_idx:
    word = wv.index2word[i]
    tf = wv.vocab[word].count
    norm = norms[i]
    ax.scatter([tf], [norm], alpha=1, edgecolors='black', s=30, label=word)

ax.grid(True, which='both')
plt.ylim(0, 30)
ax.legend()
fig.tight_layout()
fig.savefig('data/ng_norm-tf.png')
fig.show()


#%% def plot_standard_vec_norms():
norms, tfs = calc_norms(standard_vec)
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText Norm - TF')
plt.xlabel(tf_label)
plt.xscale('log')
plt.ylabel('standard norm (Gensim)')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label=fasttext_model_vocab_label)

common_words_norm, common_words_tfs = common_words_norms(standard_vec)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label=mit_10k_common_label)

sorted_idxs = matutils.argsort(norms, reverse=True)

rnd_word_idx = [
    sorted_idxs[400000], sorted_idxs[800000], sorted_idxs[1200000], sorted_idxs[1600000], sorted_idxs[1800000]
    # select_word_index(70_000, 100_000, 2.4, 2.71),
    # select_word_index(55_000, 70_000, 0.53, 0.6),
    # select_word_index(4600_000, 7000_000, 0.44, 0.47),
    # select_word_index(4600_000, 7000_000, 1.26, 1.3),
    # select_word_index(4600_000, 7000_000, 2.4, 2.71)
]
for i in rnd_word_idx:
    word = wv.index2word[i]
    tf = wv.vocab[word].count
    norm = norms[i]
    ax.scatter([tf], [norm], alpha=1, edgecolors='black', s=30, label=word)

ax.grid(True, which='both')
# plt.ylim(0, 40)
ax.legend()
fig.tight_layout()
fig.savefig('data/standard_norm-tf.png')
fig.show()


#%% def plot_no_ngram():
norms, tfs = calc_norms(no_ngram_vector)
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText Word Whole Word Token (no ngram) - TF')
plt.xlabel(tf_label)
plt.xscale('log')
plt.ylabel('norm of the whole words without sub-ngrams')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label=fasttext_model_vocab_label)

common_words_norm, common_words_tfs = common_words_norms(no_ngram_vector)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label=mit_10k_common_label)

sorted_idxs = matutils.argsort(norms, reverse=True)

rnd_word_idx = [
    sorted_idxs[400000], sorted_idxs[800000], sorted_idxs[1200000], sorted_idxs[1600000], sorted_idxs[1800000]
    # select_word_index(70_000, 100_000, 2.4, 2.71),
    # select_word_index(55_000, 70_000, 0.53, 0.6),
    # select_word_index(4600_000, 7000_000, 0.44, 0.47),
    # select_word_index(4600_000, 7000_000, 1.26, 1.3),
    # select_word_index(4600_000, 7000_000, 2.4, 2.71)
]
for i in rnd_word_idx:
    word = wv.index2word[i]
    tf = wv.vocab[word].count
    norm = norms[i]
    ax.scatter([tf], [norm], alpha=1, edgecolors='black', s=30, label=word)

ax.grid(True, which='both')
plt.ylim(0, 40)
ax.legend()
fig.tight_layout()
fig.savefig('data/no_ngram_norm-tf.png')
fig.show()


# %% def plot_histogram_of_common_and_vocab_ng_norms():
norms, _ = calc_norms(custom_vec)
print(f'vecs norms avg: {np.average(norms[np.isfinite(norms)], axis=0)}')
print(f'norms: {norms[0:10].tolist()}')
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText NG_Norm Distribution')
plt.ylabel('Word Count Density')
plt.xlabel(tf_label)
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
bins = np.linspace(0, 10, 100)
ax.hist(norms, bins=bins, alpha=0.5, label=fasttext_model_vocab_label, density=True)
common_norms, _ = common_words_norms(custom_vec)
print(f'common vecs norms avg: {np.average(common_norms[np.isfinite(common_norms)], axis=0)}')
ax.hist(common_norms, bins=bins, alpha=0.5, label=mit_10k_common_label, density=True)
ax.grid(True, which='both')
ax.legend()
fig.tight_layout()
fig.savefig('data/ng_norm-hist.png')
fig.show()


#%% calc_and_store_ng_norm_density_histogram():
density_histogram, bins = common_word_norm_density_histogram()
# np.savetxt('data/hist-probability.txt', probability_histogram)
ng_norms = pd.Series(bins).rolling(window=2).mean().iloc[1:].values
pd.DataFrame({'density': density_histogram, 'ng_norm': ng_norms}).to_csv('data/ng-norm-density-hist.csv', index=False)
np.savetxt('data/hist-bins.txt', bins)


#%%
def run_plot_density_histogram():
    pdf_df = pd.read_csv('data/ng-norm-density-hist.csv')
    density_histogram = pdf_df['density'].values
    ng_norms = pdf_df['ng_norm']

    bin_width = ng_norms[1] - ng_norms[0]
    fitfunc = lambda mu, sigma, df, x: t.pdf(x, df, mu, sigma)
    errfunc = lambda p, x, y: fitfunc(p[0], p[1], p[2], x) - y

    mean = np.sum(density_histogram * bin_width * ng_norms)
    sigma = np.sqrt(np.sum(density_histogram * bin_width * (ng_norms - mean) ** 2))
    starting_param = np.array([mean, sigma, 3])
    print(f'starting p {starting_param}')
    p, success = leastsq(errfunc, starting_param, args=(ng_norms, density_histogram), full_output=False)
    print(f'fitted_density params: {p}, success value: {success}')
    fitted_density = fitfunc(p[0], p[1], p[2], ng_norms)

    fig: Figure = plt.figure()
    plt.title(mit_10k_common_label + 'FastText NG-Norm Density Histogram')
    plt.ylabel('Probability Density')
    plt.xlabel(ng_norm_label)
    ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
    ax.bar(ng_norms, density_histogram, label='original distribution', color='grey', alpha=1, width=bin_width, edgecolor='grey')
    # ax.plot(X, fitted_density, label='fitted_density', color='green', alpha=0.5, width=bin_width) #, linestyle='--')
    fit_label = f'fitted t-distribution (df: {np.round(p[2], 1)}, mean: {np.round(p[0], 1)}, var: {np.round(p[1], 1)})'
    ax.plot(ng_norms, fitted_density, label=fit_label, color='orange', alpha=1, linestyle='--')
    ax.grid(True, which='both')
    ax.legend()
    fig.savefig('data/hist-norm.png')
    fig.show()


run_plot_density_histogram()


#%% print_word_separation
def word_split_probability(text: str):
    bin_width = bins[1] - bins[0]
    text = text.lower()
    df = pd.DataFrame(index=range(1, len(text)), columns=['word1', 'word2', 'norm1', 'norm2', 'prob1', 'prob2', 'prob'])
    for i in df.index:
        word1 = text[:i]
        word2 = text[i:]
        df.loc[i, 'word1'] = word1
        df.loc[i, 'word2'] = word2
        df.loc[i, 'norm1'] = LA.norm(custom_vec(word1))
        df.loc[i, 'norm2'] = LA.norm(custom_vec(word2))

    df['prob1'] = density_histogram[np.digitize(df['norm1'].values, bins)] * bin_width
    df['prob2'] = density_histogram[np.digitize(df['norm2'].values, bins)] * bin_width
    df['prob'] = df['prob1'] * df['prob2']
    print(df)


pdf_df = pd.read_csv('data/ng-norm-density-hist.csv')
density_histogram = pdf_df['density'].values
ng_norms = pdf_df['ng_norm']
bins = np.loadtxt('data/hist-bins.txt')

text = 'fictionaluniverse'
word_split_probability(text)

