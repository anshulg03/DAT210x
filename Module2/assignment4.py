import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2',header = 1)
df = df[0]
# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
#
# .. your code here ..
df.GP = pd.to_numeric(df.GP, errors='coerce')


# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
#
# .. your code here ..
df = df.dropna(axis = 0, thresh = 4)
df = df.drop('RK',1)
df = df.dropna()

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..


# TODO: Get rid of the 'RK' column
#
# .. your code here ..


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..



# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#
# .. your code here ..
df.dtypes
df.PCT = pd.to_numeric(df.PCT, errors='coerce')

df = df.reset_index()
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
# .. your code here ..

