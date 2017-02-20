#
# This code is intentionally missing!
# Read the directions on the course lab page!
#


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import andrews_curves
plt.style.use('ggplot')
df = pd.read_csv('C:/Users/anshangu/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data', index_col = 0)
    
#del df['area']
#del df['perimeter']


andrews_curves(df,'wheat_type')