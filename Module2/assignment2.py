import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#
# .. your code here ..
df = pd.read_csv('C:/Users/anshangu/Documents/GitHub/DAT210x/Module2/Datasets/tutorial.csv')


# TODO: Print the results of the .describe() method
#
# .. your code here ..
print(df.describe())


# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..
df.iloc[2:4,3]
df.loc[2:4,'col3']