#%% load model from disk

from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath

cap_path = datapath("/home/vackosar/src/fasttext-vector-norms-and-oov-words/data/cc.en.300.bin")
# fb_model = load_facebook_model(cap_path)
wv = load_facebook_vectors(cap_path)
wv.init_sims()
print(f'model: maxn: {wv.max_n}, minn {wv.min_n}, vocab size: {len(wv.vectors_vocab)}')


#%% util methods
import re
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from gensim import matutils
from gensim.models.utils_any2vec import ft_ngram_hashes
from math import log, sqrt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import linalg as LA
from numpy.core.multiarray import ndarray
from scipy.optimize import leastsq
from scipy.stats import t


def common_words_norms(get_vec: Callable):
    words: ndarray = pd.read_csv('data/mit-10k-words.csv', header=None, names=['word'])['word'].values
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


def common_word_norm_histogram() -> (ndarray, ndarray):
    common_norms, _ = common_words_norms(custom_vec)
    norms, _ = calc_norms(custom_vec)
    bins = np.linspace(0, 10, 300)
    norm_histogram, _ = np.histogram(norms, bins)
    common_histogram, _ = np.histogram(common_norms, bins)
    histogram = common_histogram / norm_histogram
    histogram[np.isnan(histogram)] = 0
    histogram = histogram / histogram[np.isfinite(histogram)].sum()
    return histogram, bins


def histogram_position(bins, value) -> int:
    for i, b in enumerate(bins):
        if value < b:
            if i == 0:
                raise ValueError
            else:
                return i - 1
    raise ValueError


def histogram_val(histogram: ndarray, bins: ndarray, value: float) -> float:
    return histogram[histogram_position(bins, value)]


def no_ngram_vector(word: str) -> ndarray:
    if word in wv.vocab:
        return wv.vectors_vocab[wv.vocab[word].index]

    else:
        return np.zeros(wv.vectors_vocab[0].shape[0])


#%% def plot_vec_norms():
norms, tfs = calc_norms(custom_vec)

seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText norm-tf')
plt.xlabel('tf (fasttext word count)')
plt.xscale('log')
plt.ylabel('norm')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label='all')

common_words_norm, common_words_tfs = common_words_norms(custom_vec)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label='common')

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
fig.savefig('data/figure.png')
fig.show()


#%% def plot_standard_vec_norms():
norms, tfs = calc_norms(standard_vec)
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText norm-tf')
plt.xlabel('tf (fasttext word count)')
plt.xscale('log')
plt.ylabel('norm')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label='all')

common_words_norm, common_words_tfs = common_words_norms(standard_vec)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label='common')

sorted_idxs = matutils.argsort(norms, reverse=True)
# rnd_word_idx = [sorted_idxs[400000], sorted_idxs[800000], sorted_idxs[1200000], sorted_idxs[1600000], sorted_idxs[1800000]]

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
fig.savefig('data/standard-figure.png')
fig.show()


#%% def plot_no_ngram():
norms, tfs = calc_norms(no_ngram_vector)
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText norm-tf')
plt.xlabel('tf (fasttext word count)')
plt.xscale('log')
plt.ylabel('norm')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.scatter(tfs, norms, alpha=0.6, edgecolors='none', s=5, label='all')

common_words_norm, common_words_tfs = common_words_norms(no_ngram_vector)
ax.scatter(common_words_tfs, common_words_norm, alpha=0.8, edgecolors='none', s=5, label='common')

sorted_idxs = matutils.argsort(norms, reverse=True)
# rnd_word_idx = [sorted_idxs[400000], sorted_idxs[800000], sorted_idxs[1200000], sorted_idxs[1600000], sorted_idxs[1800000]]

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
fig.savefig('data/no-ngram-figure.png')
fig.show()


# %% def plot_histogram_norm():
norms, _ = calc_norms(custom_vec)
print(f'vecs norms avg: {np.average(norms[np.isfinite(norms)], axis=0)}')
# norms = np.apply_over_axes(lambda w: LA.norm(wv.word_vec(w)), 1, words)
# norms = np.apply_along_axis(LA.norm, 0, vectors_vocab)
# norms = np.apply_along_axis(LA.norm, 1, vectors_vocab)
print(f'norms: {norms[0:10].tolist()}')
seaborn.set(style='white', rc={'figure.figsize': (12, 8)})
fig: Figure = plt.figure()
plt.title('FastText norm-tf')
plt.xlabel('tf (fasttext word count)')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
bins = np.linspace(0, 10, 100)
ax.hist(norms, bins=bins, alpha=0.5, label='fasttext', density=True)
common_norms, _ = common_words_norms(custom_vec)
print(f'common vecs norms avg: {np.average(common_norms[np.isfinite(common_norms)], axis=0)}')
ax.hist(common_norms, bins=bins, alpha=0.5, label='common', density=True)
ax.grid(True, which='both')
ax.legend()
fig.tight_layout()
fig.savefig('data/hist.png')
fig.show()


#%% def fit_gaussian():
fitfunc = lambda mu, sigma, x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
gaussian = lambda x: fitfunc(30, 2, x)  #3*np.exp(-(30-x)**2/20.)
X = np.arange(100)
Y = gaussian(X)
# X = np.arange(data.size)
# mu = np.sum(X*data)/np.sum(data)
# sigma = np.sqrt(np.abs(np.sum((X-mu)**2*data)/np.sum(data)))
# max = data.max()
# Target function
# Initial guess for the first set's parameters
# p1 = r_[-15., 0., -1.]
# Initial guess for the second set's parameters
# p2 = r_[-15., 0., -1.]
# Initial guess for the common period
# T = 0.8
# Vector of the parameters to fit, it contains all the parameters of the problem, and the period of the oscillation is not there twice !
# p = r_[T, p1, p2]
# Cost function of the fit, compare it to the previous example.
errfunc = lambda p, x, y: fitfunc(p[0], p[1], x) - y
# This time we need to pass the two sets of data, there are thus four "args".
starting_param = np.array([20, 1])
p, success = leastsq(errfunc, starting_param, args=(X, Y))
print(f'p {p}, succ {success}')
# plt.plot(X, Y)
fig: Figure = plt.figure()
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.plot(X, gaussian(X))
ax.plot(X, fitfunc(p[0], p[1], X))
plt.show()


#%% print_word_separation
# density, bins = common_word_norm_histogram()
# np.savetxt('data/hist-density.txt', density)
# np.savetxt('data/hist-bins.txt', bins)
density = np.loadtxt('data/hist-density.txt')
bins = np.loadtxt('data/hist-bins.txt')
X = pd.Series(bins).rolling(window=2).mean().iloc[1:].values
pd.DataFrame({'density': density, 'X': X}).to_csv('data/hist-norm.csv', index=False)
# X = [x for x in iter(bins)]

bin_width = bins[-1] / (bins.shape[0] - 1)
# fitfunc = lambda mu, sigma, x: bin_width * 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- 1 / 2 * ((x - mu) / sigma) ** 2)
#fitfunc = lambda mu, sigma, x: bin_width * norm.pdf(x, mu, sigma)
fitfunc = lambda mu, sigma, df, x: bin_width * t.pdf(x, df, mu, sigma)
errfunc = lambda p, x, y: fitfunc(p[0], p[1], p[2], x) - y

mean = np.sum(density * X)
sigma = np.sqrt(np.sum(density * (X - mean) ** 2))
starting_param = np.array([mean, sigma, 2])
# starting_param = np.array([2.5, 0.5])
print(f'starting p {starting_param}')
p, success = leastsq(errfunc, starting_param, args=(X, density))
fitted = fitfunc(p[0], p[1], p[2], X)
# fitted = fitfunc(mean, sigma, X)
print(f'p {p}, succ {success}')

fig: Figure = plt.figure()
# plt.title('FastText norm-tf')
# plt.xlabel('tf (fasttext word count)')
ax: Axes = fig.add_subplot(1, 1, 1) #axisbg="1.0")
ax.bar(X, density, label='probability', color='grey', alpha=1, width=bin_width)
# ax.plot(X, fitted, label='fitted', color='green', alpha=0.5, width=bin_width) #, linestyle='--')
ax.plot(X, fitted, label='fitted', color='orange', alpha=1, linestyle='--')
ax.grid(True, which='both')
ax.legend()
fig.savefig('data/hist-norm.png')
fig.show()

text = 'accessorykit'
print(f'orig norm: {LA.norm(wv.word_vec(text))}')
for i in range(1, len(text)):
    word_1 = text[:i].lower()
    word_2 = text[i:].lower()
    norm1 = LA.norm(custom_vec(word_1))
    norm2 = LA.norm(custom_vec(word_2))
    # norm1 = LA.norm(wv.word_vec(word_1))
    # norm2 = LA.norm(wv.word_vec(word_2))
    prob1 = histogram_val(density, bins, norm1)
    prob2 = histogram_val(density, bins, norm2)
    print(f'norm {word_1} {norm1}, norm2 {word_2} {norm2}, sum {(norm1 + norm2) / 2}, squared {(sqrt(norm1*norm1 + norm2*norm2) / 2)}, prob1 {prob1} prob2 {prob2}, prob {prob1 * prob2}')

print(f'1 {LA.norm(wv.vectors_vocab[wv.vocab["congratulations"].index])}')
print(f'1 {LA.norm(wv.get_vector("congratulations"))}')
print(f'2 {LA.norm(wv.vectors_vocab[wv.vocab["videos"].index])}')
print(f'2 {LA.norm(wv.get_vector("videos"))}')

