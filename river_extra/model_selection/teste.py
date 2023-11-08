import hyanova

import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm




df = pd.read_csv('/Users/brunoveloso/Downloads/ensaio11.csv', names=['hp1', 'hp2', 'hp3', 'instances', 'score'])
print(df)



for i in df.instances.unique():
    print('------->' + str(i))
    df3=df[df['instances']==i]
    df3=df3.reset_index()
    df2, params = hyanova.read_df(df3, 'score')
    #print(df2['score'].unique())
    importance = analyze_incr(df2,max_iter=-1)
    print(importance)
    #break

