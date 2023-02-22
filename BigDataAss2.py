import pandas as pd
import numpy as np


print("\n\n--------Task 1:--------")

# Load data from the provided stock_data.csv file
stock_data_file = "stock_data.csv"

stock_data = pd.read_csv(stock_data_file, sep=',', header=0)
print(stock_data)



print("\n\n--------Task 2:-------- ")

# Sorted data by names, other collumns are irrelevant, so we omit them and duplicates disappear
set_of_names = sorted(set(stock_data['Name'].tolist())) 

# Number of Unique names
num_of_unique_names = len(set_of_names)
print("\nNumber of unique names: ")
print (num_of_unique_names)

# First 5 names (alphabetically sorted)
first_five_names = set_of_names[:5]
print("\nFirst five (sorted) names are: ")
print(first_five_names)



print("\n\n--------Task 3:-------- ")

# Convert the data in the column-date into datetime type
stock_data['date'] = pd.to_datetime(stock_data['date'])

# Convert the 1st of July 2014 to datatime type
began_after = pd.to_datetime('2014-07-01')

# Convert the 31st of December 2017 to datatime type
finished_before = pd.to_datetime('2017-12-31')

# Get all the names and first data of their appearance (date.idxmin)
first_appearance = stock_data.loc[stock_data.groupby('Name')['date'].idxmin()]
# Names with their first appearance after 1/07/2014. Changed to list so their id isn't displayed
began_after_to_remove = first_appearance[first_appearance['date'] > began_after]['Name'].tolist()

# Get all the names and last data of their appearance (date.idxmax)
last_appearance = stock_data.loc[stock_data.groupby('Name')['date'].idxmax()]
# Names with their last appearance before 31/12/2017. Changed to list so their id isn't displayed
finished_before_to_remove = last_appearance[last_appearance['date'] < finished_before]['Name'].tolist()

# Concat the names that are meant to be deleted and remove duplicates
to_remove = pd.unique(began_after_to_remove + finished_before_to_remove)
print("\nNames that were removed:")
print(to_remove)

# Remove the names to_remove from the list of all the names. sort it
set_of_names_updated = sorted([i for i in set_of_names if i not in to_remove])
# Update the set of all the names
set_of_names = set_of_names_updated
print("\nThat many unique names are left:")
print(len(set_of_names))

# Update the dataframe
is_in_it = stock_data['Name'].isin(set_of_names)
stock_data = stock_data[is_in_it]



print("\n\n--------Task 4:-------- ")

# Date range between 2014-07-01 and 2017-12-31
stock_data = stock_data[stock_data['date'] >= began_after]
stock_data = stock_data[stock_data['date'] <= finished_before]

# I need to find that many measures on one day
num_of_names = len(set_of_names)

# Data on which all the names were registered and are in date range constraints
data_true_false = stock_data.groupby('date')['date'].count().reset_index(name="count")
data_true_false = data_true_false.loc[data_true_false['count'] == num_of_names]

# How many rows = how many days
print("\nThat many dates are left:")
print(len(data_true_false))

# Save it for task 5
save_data = data_true_false['date'].tolist()
# Take only date column and convert it to date type, to list
data_true_false = data_true_false['date'].dt.date.tolist()

# Display first 5 and last 5 days
print("\nFirst 5 dates:")
print("{one_}, {two_}, {three_}, {four_}, {five_}".format(one_=data_true_false[0], two_=data_true_false[1], three_=data_true_false[2], four_=data_true_false[3], five_=data_true_false[4]))
print("\nLast 5 dates:")
print("{one_}, {two_}, {three_}, {four_}, {five_}".format(one_=data_true_false[-1], two_=data_true_false[-2], three_=data_true_false[-3], four_=data_true_false[-4], five_=data_true_false[-5]))



print("\n\n--------Task 5:-------- ")

# New dataframe with names as columns and dates as rows
new_df = pd.DataFrame(index=save_data, columns=set_of_names)

# Function that sets up specific cell value to linked close value from the given data frame 
def set_close_values(x):
      new_df.loc[x['date'],x['Name']] = x['close']
      return 0

# Transform stock_data so that it only consists of names and dates that we have in newly created dataframe
df_to_work_on = stock_data[(stock_data['Name'].isin(set_of_names)) & (stock_data['date'].isin(save_data))]
# Apply the function (set_close_values) described above, using the data from dataframe in the line above
df_to_work_on.apply(lambda x: set_close_values(x), axis=1)

# Display first and last 5 rows
print("\nFirst 5 rows:")
print(new_df.head(5))
print("\nLast 5 rows:")
print(new_df.tail(5))


## ALTERNATIVE SOLUTION ##
# Takes too long time to compute
# for every name, for every date, select that name's, date's close price from stock_data and add it to new df
'''
for x in set_of_names:
      for i in save_data:
            hold = stock_data[(stock_data['Name'] == x) & (stock_data['date'] == i)]['close'].tolist()
            new_df.loc[i,x] = hold[0]
'''


print("\n\n--------Task 6:-------- ")

# Percentage change between the current and a prior element
new_df2 = new_df.pct_change(periods=1)
# Drop the first row with NaN values
new_df2 = new_df2.iloc[1: , :]

# Display first and last 5 rows
print("\nFirst 5 rows:")
print(new_df2.head(5))
print("\nLast 5 rows:")
print(new_df2.tail(5))



print("\n\n--------Task 7:-------- ")

from sklearn.decomposition import PCA

# Material from the labs
pca = PCA()
pca.fit(new_df2)

# Principal components:
print("\nTop 5 Principal components with the largest eigenvalues:")
print(pca.components_[:5])


# There are 'pca.n_components_' components



'''
### If I was meant to print top 5 eigenvalues, then they are here: ###
# Return only eigenvalues
eigenvalues = np.linalg.eigvalsh(pca.components_)
////////eigValue, _ = np.linalg.eig(pc.components_)

# The eigenvector with the largest eigenvalue is the direction with most variability. 
# We call this eigenvector the first principle component (or axis). So 1st principal - 1st eigenvalue, 2-2, 3-3 and so on
# eigen_vectors = pd.DataFrame(pca.components_)

# Principle components:
print("\nTop 5 Principal components and their corresponding Eigenvalue:")
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 1, eigen_ = eigenvalues[-1]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 2, eigen_ = eigenvalues[-2]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 3, eigen_ = eigenvalues[-3]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 4, eigen_ = eigenvalues[-4]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 5, eigen_ = eigenvalues[-5]))
'''


print("\n\n--------Task 8:-------- ")

# Extract the explained variance ratios 
# pca.explained_variance_ratio_

# To explain what percentage of variance is explained by the first -
# principal component, I have to multiply the first principal element by 100
print("\nThe percentage of variance the first principal component explains:")
print (pca.explained_variance_ratio_[0]*100)


# Plot the first 20 explained variance ratios
import matplotlib.pyplot as plt

# Numbers 1 - 20
x = range(20)
# First 20 explained variance ratios
y = pca.explained_variance_ratio_[:20]

# Plotting the points
plt.plot(x, y, color='green', linewidth = 2)
# Changing the tick between the points, so data is better visible
plt.xticks(range(0,21,1))
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('Explained variance ratios')
# giving a title to my graph
plt.title('First 20 Explained variance ratios Plot')

# function to show the plot
plt.show()
print("\nPlot has been opened in a new window")
# Close the plot so I can create a new one later
plt.close()



print("\n\n--------Task 9:-------- ")

# Calculate the cumulative variance ratios using numpy.cumsum
cum_var_ratios = np.cumsum(pca.explained_variance_ratio_)

x = range(pca.n_components_)
y = cum_var_ratios

above_95 = 0
# Set above_95 to the first principal components for which cumulative variance ratio >= 0.95
for i in range(len(x)):
      if y[i] >= 0.95:
            above_95 = x[i]
            break;

# # Plot all the cumulative variance ratios (x axis = principal component, y axis = cumulative variance ratio).
plt.plot(x, y, color='green', linewidth = 2, label = "plot")

# Marking the Principal components for which the cumulative variance ratio >= 0.95
x2 = [above_95, x[pca.n_components_ - 1]]
y2 = [x[0],x[0]]
plt.plot(x2, y2, color='red', linewidth = 3, label = "marked components")

# Additional line to illustrate it better
x3 = [above_95, above_95]
y3 = [x[0], 1]
plt.plot(x3, y3, color='blue', linewidth = 1, label = "<95%<=", linestyle="--")


# Show the labels of plots
plt.legend()
# naming the x axis
plt.xlabel('principal component')
# naming the y axis
plt.ylabel('cumulative variance ratio')
# giving a title to my graph
plt.title('Plot 2 - Principal/Cumulative')

# function to show the plot
plt.show()
print("\nPlot has been opened in a new window")
# Close the plot so I can create a new one later
plt.close()



## ALTERNATIVE SOLUTION ##
## Here I marked points on the plot for Principals for which Cumulative >= 0.95  ##
'''
print("\n\n--------Task 9:-------- ")

# Calculate the cumulative variance ratios using numpy.cumsum
cum_var_ratios = np.cumsum(pca.explained_variance_ratio_)

x = pca.explained_variance_ratio_
y = cum_var_ratios

list_above_95 = []
# Add to the list positions of the principle components for which cumu;ative variance ration >= 0.95
for i in range(len(x)):
      if y[i] >= 0.95:
            list_above_95.append(i)

# # Plot all the cumulative variance ratios (x axis = principal component, y axis = cumulative variance ratio).
plt.plot(x, y, color='green', linewidth = 2)
# Marking points for which cumulative variance ratio is >= 95%
plt.plot(x, y, markevery=list_above_95, ls = "", marker="o", markerfacecolor='blue', markersize=3, label="points")
# naming the x axis
plt.xlabel('principal component')
# naming the y axis
plt.ylabel('cumulative variance ratio')
# giving a title to my graph
plt.title('Plot 2 - Principal/Cumulative')

# function to show the plot
plt.show()
print("\nPlot has been opened in a new window")
# Close the plot so I can create a new one later
plt.close()
'''


print("\n\n--------Task 10:-------- ")

############### Normalise your dataframe from step (6) so that the columns have zero mean and unit variance #########

normalized_df = new_df2

for column in normalized_df.columns:
      normalized_df[column] = (normalized_df[column]-normalized_df[column].mean())  / normalized_df[column].std()

# I will copy steps from 7-9.
# Everything I change will be the dataframe that I operate on. The line """pca.fit(normalized_df)"""


print("\n\n--------Task 10 - Step 7:-------- ")


# Material from the labs
pca = PCA()
pca.fit(normalized_df)

# Principal components:
print("\nTop 5 Principal components with the largest eigenvalues:")
print(pca.components_[:5])

# There are 'pca.n_components_' components



'''
### If I was meant to print top 5 eigenvalues, then they are here: ###
# Return only eigenvalues
eigenvalues = np.linalg.eigvalsh(pca.components_)

# The eigenvector with the largest eigenvalue is the direction with most variability. 
# We call this eigenvector the first principal component (or axis). So 1st principal - 1st eigenvalue, 2-2, 3-3 and so on
# eigen_vectors = pd.DataFrame(pca.components_)

# Principle components:
print("\nTop 5 Principle components and their corresponding Eigenvalue:")
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 1, eigen_ = eigenvalues[-1]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 2, eigen_ = eigenvalues[-2]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 3, eigen_ = eigenvalues[-3]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 4, eigen_ = eigenvalues[-4]))
print("\nPrincipal component: {prin_} its eigenvalue: {eigen_}".format(prin_ = 5, eigen_ = eigenvalues[-5]))
'''


print("\n\n--------Task 10 - Step 8:-------- ")

# Extract the explained variance ratios 
# pca.explained_variance_ratio_

# To explain what percentage of variance is explained by the first -
# principal component, I have to multiply the first principal element by 100
print("\nThe percentage of variance the first principal component explains:")
print (pca.explained_variance_ratio_[0]*100)


# Plot the first 20 explained variance ratios
import matplotlib.pyplot as plt

# Numbers 1 - 20
x = range(20)
# First 20 explained variance ratios
y = pca.explained_variance_ratio_[:20]

# Plotting the points
plt.plot(x, y, color='green', linewidth = 2)
# Changing the tick between the points, so data is better visible
plt.xticks(range(0,21,1))
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('Explained variance ratios')
# giving a title to my graph
plt.title('First 20 Explained variance ratios Plot')

# function to show the plot
plt.show()
print("\nPlot has been opened in a new window")
# Close the plot so I can create a new one later
plt.close()



print("\n\n--------Task 10 - Step 9:-------- ")

# Calculate the cumulative variance ratios using numpy.cumsum
cum_var_ratios = np.cumsum(pca.explained_variance_ratio_)

x = range(pca.n_components_)
y = cum_var_ratios

above_95 = 0
# Set above_95 to the first principal components for which cumulative variance ratio >= 0.95
for i in range(len(x)):
      if y[i] >= 0.95:
            above_95 = x[i]
            break;

# # Plot all the cumulative variance ratios (x axis = principal component, y axis = cumulative variance ratio).
plt.plot(x, y, color='green', linewidth = 2, label = "plot")

# Marking the Principal components for which the cumulative variance ratio >= 0.95
x2 = [above_95, x[pca.n_components_ - 1]]
y2 = [x[0],x[0]]
plt.plot(x2, y2, color='red', linewidth = 3, label = "marked components")

# Additional line to illustrate it better
x3 = [above_95, above_95]
y3 = [x[0], 1]
plt.plot(x3, y3, color='blue', linewidth = 1, label = "<95%<=", linestyle="--")


# Show the labels of plots
plt.legend()
# naming the x axis
plt.xlabel('principal component')
# naming the y axis
plt.ylabel('cumulative variance ratio')
# giving a title to my graph
plt.title('Plot 2 - Principal/Cumulative')

# function to show the plot
plt.show()
print("\nPlot has been opened in a new window")
# Close the plot so I can create a new one later
plt.close()

print("\nOPERATIONS FINISHED WITH A SUCCESS!!!\n")


