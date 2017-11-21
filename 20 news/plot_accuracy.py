import matplotlib.pyplot as plt
import numpy as np

topic_num = np.array([5, 10, 20, 25, 30, 40, 50])
title_font = {'fontname': 'Arial', 'size': 10, 'color': 'black', 'weight': 'normal'}
plt.figure(1)

## labeled data = 5
plt.subplot(221)
SVD_5 = np.array([15.55, 25.39, 31.00, 32.07, 32.24, 34.03, 33.58])
LDA_5 = np.array([33.35, 33.96, 33.80, 33.89, 34.26, 34.07, 33.54])
plt.title('number of labeled documents = 5', **title_font)
plt.xlabel('number of topics', **title_font)
plt.ylabel('accuracy (%)', **title_font)
plt.xlim((0, 55))
plt.xticks(topic_num, topic_num, **title_font)
plt.ylim((10, 60))
plt.yticks(range(10, 61, 5), range(10, 61, 5), **title_font)
plt.plot(topic_num, SVD_5, 'r.-', label='SVD')
plt.plot(topic_num, LDA_5, 'b.-', label='LDA')
plt.legend(loc='lower right', fontsize=6)

## labeled data = 10
plt.subplot(222)
SVD_10 = np.array([18.33, 30.21, 37.86, 40.66, 40.06, 42.39, 42.44])
LDA_10 = np.array([42.62, 41.80, 42.70, 42.96, 42.91, 43.24, 42.32])
plt.title('number of labeled documents = 10', **title_font)
plt.xlabel('number of topics', **title_font)
plt.ylabel('accuracy (%)', **title_font)
plt.xlim((0, 55))
plt.xticks(topic_num, topic_num, **title_font)
plt.ylim((10, 60))
plt.yticks(range(10, 61, 5), range(10, 61, 5), **title_font)
plt.plot(topic_num, SVD_10, 'r.-', label='SVD')
plt.plot(topic_num, LDA_10, 'b.-', label='LDA')
plt.legend(loc='lower right', fontsize=6)

## labeled data = 15
plt.subplot(223)
SVD_15 = np.array([20.08, 32.88, 41.99, 44.21, 45.65, 47.25, 48.78])
LDA_15 = np.array([48.38, 48.31, 48.53, 48.87, 48.55, 47.94, 48.63])
plt.title('number of labeled documents = 15', **title_font)
plt.xlabel('number of topics', **title_font)
plt.ylabel('accuracy (%)', **title_font)
plt.xlim((0, 55))
plt.xticks(topic_num, topic_num, **title_font)
plt.ylim((10, 60))
plt.yticks(range(10, 61, 5), range(10, 61, 5), **title_font)
plt.plot(topic_num, SVD_15, 'r.-', label='SVD')
plt.plot(topic_num, LDA_15, 'b.-', label='LDA')
plt.legend(loc='lower right', fontsize=6)

## labeled data = 20
plt.subplot(224)
SVD_20 = np.array([20.67, 34.78, 44.60, 46.97, 48.28, 51.02, 52.73])
LDA_20 = np.array([52.63, 52.93, 53.07, 52.92, 52.74, 52.82, 52.01])
plt.title('number of labeled documents = 20', **title_font)
plt.xlabel('number of topics', **title_font)
plt.ylabel('accuracy (%)', **title_font)
plt.xlim((0, 55))
plt.xticks(topic_num, topic_num, **title_font)
plt.ylim((10, 60))
plt.yticks(range(10, 61, 5), range(10, 61, 5), **title_font)
plt.plot(topic_num, SVD_20, 'r.-', label='SVD')
plt.plot(topic_num, LDA_20, 'b.-', label='LDA')
plt.legend(loc='lower right', fontsize=6)

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.4, wspace=0.35)
plt.savefig('accuracy.pdf')