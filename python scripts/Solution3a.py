import ppscore as pps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')

data = pd.read_csv('dataset_challenge_one.tsv', delimiter='\t')
data = data.fillna(0)
predictors = data.columns[:-1]
score = []
var = []
for i in predictors:
    score.append(pps.score(data, i, 'class')['ppscore'])
    var.append(i)

ppscore_df = pd.DataFrame({'predictor': var, 'ppscore': score})
ppscore_df = ppscore_df.sort_values(['ppscore'],
                                    ascending=False).reset_index()
plt.figure(figsize=(6, 3))
plt.plot(ppscore_df['predictor'].index, ppscore_df['ppscore'])
plt.grid()
plt.xticks(np.arange(0, 1600, 50), rotation='vertical', fontsize=10)
plt.ylabel('ppscore', fontsize=16)
plt.xlabel('Predictor', fontsize=10)
plt.yticks(np.arange(0, 0.3, 0.05), fontsize=10)
plt.title('Predictive Power Score Plot', fontsize=20)
plt.show()

ppscore_df['ppscore'].value_counts()
bins = [
    -1,
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    ]
labels = [
    '< 0',
    '0-0.05',
    '0.05-0.1',
    '0.1-0.15',
    '0.15-0.2',
    '0.2-0.25',
    '0.25-0.3',
    ]
ppscore_df['binned'] = pd.cut(ppscore_df['ppscore'], bins,
                              labels=labels)
plt.bar(ppscore_df['binned'].value_counts().index, 
        ppscore_df['binned'].value_counts())
plt.title('Count of variables for ppscore bins', fontsize=15)
plt.ylabel('Count of variables', fontsize=12)
plt.xlabel('PPScore', fontsize=12)
plt.xticks(rotation=45)
plt.show()

