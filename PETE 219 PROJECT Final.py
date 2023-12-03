#!/usr/bin/env python
# coding: utf-8

# # WELL LOG DATA

# 1 and 2, data import and cleaning. data visualization

# In[1]:

import lasio as las
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import missingno as msno

df = las.read('1051661161.las')
df = df.df()

df['DEPTH'] = df.index
index = list(range(5541))
df['Index'] = index
df.set_index('Index', inplace = True)


# In[2]:


df.describe()


# In[31]:


df['GR'][df['GR'] > 500] = np.nan
df['GR'][df['GR'] < 0] = np.nan

df['RILM'][df['RILM'] > 8500] = np.nan

df['RILD'][df['RILD'] > 2500] = np.nan

msno.matrix(df)


# In[4]:


#classic logs of GR, RILD, RILM, RLL3 and SP


# Replace common placeholder values with NaN and interpolate
df.replace(-999.25, np.nan, inplace=True)
df.interpolate(method='linear', inplace=True)

# assign the column to a variable for easier reading
curve = df['GR']

left_col_value = 0
right_col_value = 300

# calculate the span of values
span = abs(left_col_value - right_col_value)

#assign a color map
cmap = plt.get_cmap('spring_r')

#create array of values to divide up the area under curve
color_index = np.arange(left_col_value, right_col_value, span / 100)

#setup the plot
ax = df.plot(x='GR', y='DEPTH', c='black', lw=0.5, legend=False, figsize=(7,10))

plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.ylabel("Depth (ft)")
plt.title('Gamma Ray Plot with No Fill')
ax.set_xlabel("GR (API)")
ax.xaxis.set_label_position('top') 

#loop through each value in the color_index
for index in sorted(color_index):
    index_value = (index - left_col_value)/span
    color = cmap(index_value) #obtain colour for color index value
    plt.fill_betweenx(df['DEPTH'], curve, where = curve >= index,  color = color)

# Reverse the y-axis (depth should increase downwards)
ax.set_ylim(max(df.index), min(df.index))
plt.ylim(3000, 195)
plt.xlim(0, 250)
# Set labels and title
ax.set_xlabel('Gamma Ray (API)')
ax.set_ylabel('Depth')
ax.set_title('Gamma Ray Log with Shaded Areas')
ax.grid(True)

# Show the plot
plt.show()


# In[5]:


#logs of other properties

fig, axes = plt.subplots(nrows=1, ncols=4, constrained_layout = True)

# populate subplot 2 with RLL3
ax1 = df.plot(ax=axes[0], x ='RLL3', y='DEPTH', c='red', lw=1, legend=False, figsize=(7,10), ylim=[3000, 195], title= "shallow res (ohm-m)")
ax1.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

# populate subplot 3 with RILM
ax2 = df.plot(ax=axes[1], x ='RILM', y='DEPTH', c='green', lw=1, legend=False, figsize=(7,10), ylim=[3000, 195], title= "medium res (ohm-m)")
ax2.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

# populate subplot 4 with RLLD
ax3 = df.plot(ax=axes[2], x ='RILD', y='DEPTH', c='blue', lw=1, legend=False, figsize=(7,10), ylim=[3000, 195], title= "deep res (ohm-m)")
ax3.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

# populate subplot 4 with RLLD
ax4 = df.plot(ax=axes[3], x ='SP', y='DEPTH', c='black', lw=1, legend=False, figsize=(7,10), ylim=[3000, 195], title= "sp (mV)")
ax4.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)


# In[6]:


#histograms with mean and standard dev

#MEDIUM RES HISTOGRAM

meanRILM = df['RILM'].mean()
p5_RILM = df['RILM'].quantile(0.05)
p95_RILM = df['RILM'].quantile(0.95)
medianRILM = df['RILM'].median()

plotRILM = plt.hist(df['RILM'], 150, edgecolor = 'black')
plt.xlabel('Medium Res')
plt.ylabel('Frequency')
plt.title('Medium res Histogram')
plt.axvline(meanRILM, color = 'red', label = 'mean')
plt.axvline(p5_RILM, color = 'green', label = '5th percentile')
plt.axvline(p95_RILM, color = 'orange', label = '95th percentile')
plt.axvline(medianRILM, color = 'blue', label = 'median')
plt.xlim(0,120)
plt.legend()
plt.show(plotRILM)

#GAMMA RAY HISTOGRAM

meanGR = df['GR'].mean()
p5_GR = df['GR'].quantile(0.05)
p95_GR = df['GR'].quantile(0.95)
medianGR = df['GR'].median()

plotGR = plt.hist(df['GR'], 150, edgecolor = 'black')
plt.xlabel('Gamma ray')
plt.ylabel('Frequency')
plt.title('Gamma Ray Histogram')
plt.axvline(meanGR, color = 'red', label = 'mean')
plt.axvline(p5_GR, color = 'green', label = '5th percentile')
plt.axvline(p95_GR, color = 'orange', label = '95th percentile')
plt.axvline(medianGR, color = 'blue', label = 'median')
plt.xlim(0,500)
plt.legend()
plt.show(plotGR)

#SHALLOW RES HISTOGRAM

meanRLL3 = df['RLL3'].mean()
p5_RLL3 = df['RLL3'].quantile(0.05)
p95_RLL3 = df['RLL3'].quantile(0.95)
medianRLL3 = df['RLL3'].median()

plotRLL3 = plt.hist(df['RLL3'], 150, edgecolor = 'black')
plt.xlabel('Shallow Res')
plt.ylabel('Frequency')
plt.title('Shallow res Histogram')
plt.axvline(meanRLL3, color = 'pink', label = 'mean')
plt.axvline(p5_RLL3, color = 'maroon', label = '5th percentile')
plt.axvline(p95_RLL3, color = 'orange', label = '95th percentile')
plt.axvline(medianRLL3, color = 'red', label = 'median')
plt.xlim(0,200)
plt.legend()
plt.show(plotRLL3)

#DEEP RES HISTOGRAM

meanRILD = df['RILD'].mean()
p5_RILD = df['RILD'].quantile(0.05)
p95_RILD = df['RILD'].quantile(0.95)
medianRILD = df['RILD'].median()

plotRILD = plt.hist(df['RILD'], 150, edgecolor = 'black')
plt.xlabel('Deep Res')
plt.ylabel('Frequency')
plt.title('Deep res Histogram')
plt.axvline(meanRILD, color = 'red', label = 'mean')
plt.axvline(p5_RILD, color = 'green', label = '5th percentile')
plt.axvline(p95_RILD, color = 'orange', label = '95th percentile')
plt.axvline(medianRILD, color = 'blue', label = 'median')
plt.xlim(0,120)
plt.legend()
plt.show(plotRILD)


# In[42]:


fig = plt.subplots(figsize=(7,10))

ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
ax2 = ax1.twiny()

ax1.plot('RHOB', 'DEPTH', data=df, color='red', lw=0.5)
ax1.set_xlim(-2.55, 2.95)
ax1.set_xlabel('Bulk Density')
ax1.xaxis.label.set_color("red")
ax1.tick_params(axis='x', colors="red")
ax1.spines["top"].set_edgecolor("red")

ax2.plot('CNPOR', 'DEPTH', data=df, color='blue', lw=0.5)
ax2.set_xlim(80, -40)
ax2.set_xlabel('Neutron porosity')
ax2.xaxis.label.set_color("blue")
ax2.spines["top"].set_position(("axes", 1.08))
ax2.tick_params(axis='x', colors="blue")
ax2.spines["top"].set_edgecolor("blue")


for ax in [ax1, ax2]:
    ax.set_ylim(3000, 195)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")



# In[32]:
dfnew = df.drop(['AVTX', 'BVTX','CILD','CNDL','CNLS', 'CNSS','LSPD','LTEN'], axis = 1)

cor1 = dfnew.corr() # correlations as table
# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(dfnew.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Log Properties Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

X1 = df['DT']
Y1 = df['CNPOR']
plt.scatter(X1,Y1)
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()

# In[ ]:

# Function to calculate V-Shale (using Gamma Ray log)
def calculate_vshale(gamma_ray, gamma_ray_max, gamma_ray_min):
    return ((gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min))

# Add V-Shale column to DataFrame
df['VSHALE'] = calculate_vshale(df['GR'], df['GR'].max(), df['GR'].min())

# Function to calculate Effective Stress (using RHOB and Depth)
def calculate_effective_stress(rhob, depth, fluid_density=1.0):
    # Assuming fluid_density in g/cc, depth in meters, and g (gravity) = 9.81 m/s²
    return rhob * 9.81 * depth - fluid_density * 9.81 * depth

# Add Effective Stress column to DataFrame
df['EFFECTIVE_STRESS'] = calculate_effective_stress(df['RHOB'], df.index)


# # poro perm

# In[8]:


#DATA IMPORT AND CLEANING

import numpy as np

# Load the CSV file containing poro-perm data
dfp = pd.read_csv('poro_perm_data.csv')


# Basic data cleaning steps
# Identify and handle the doctored values (e.g., unrealistic porosity or permeability values)
dfp['Porosity (%)'] = dfp['Porosity (%)'].apply(lambda x: np.nan if x < 0 else x)
dfp['Permeability (mD)'] = dfp['Permeability (mD)'].apply(lambda x: np.nan if x < 0 else x)

# fill NaN values if necessary
dfp.fillna(method='bfill', inplace=True)

dfp = dfp.drop(list(range(118,120)))
display(dfp)


# In[9]:


mean_phi = dfp['Porosity (%)'].mean()
p5_phi = dfp['Porosity (%)'].quantile(0.05)
p95_phi = dfp['Porosity (%)'].quantile(0.95)
median_phi = dfp['Porosity (%)'].median()

plot_phi = plt.hist(dfp['Porosity (%)'],15, edgecolor = 'black')
plt.xlabel('Porosity (%)')
plt.ylabel('Frequency')
plt.title('Porosity (%) Histogram')
plt.axvline(mean_phi, color = 'red', label = 'mean')
plt.axvline(p5_phi, color = 'green', label = '5th percentile')
plt.axvline(p95_phi, color = 'orange', label = '95th percentile')
plt.axvline(median_phi, color = 'blue', label = 'median')
plt.xlim(0,30)
plt.legend()
plt.show(plot_phi)


# In[10]:


#Permeability

mean_perm = dfp['Permeability (mD)'].mean()
p5_perm = dfp['Permeability (mD)'].quantile(0.05)
p95_perm = dfp['Permeability (mD)'].quantile(0.95)
median_perm = dfp['Permeability (mD)'].median()

plot_perm = plt.hist(dfp['Permeability (mD)'], 25, edgecolor = 'black')
plt.xlabel('Permeability (mD)')
plt.ylabel('Frequency')
plt.title('Permeability (mD) Histogram')
plt.axvline(mean_phi, color = 'red', label = 'mean')
plt.axvline(p5_phi, color = 'green', label = '5th percentile')
plt.axvline(p95_phi, color = 'orange', label = '95th percentile')
plt.axvline(median_phi, color = 'blue', label = 'median')
plt.xlim(0,120)
plt.legend()
plt.show(plot_perm)


# In[11]:


#cross plot
import seaborn as sns
import scipy.stats as stats
# Cross plots with different markers for facies
# Ensure 'facies' is a category and then plot
if 'Facies' in dfp.columns:
    sns.scatterplot(data=dfp, x='Porosity (%)', y='Permeability (mD)', hue='Facies')
    plt.title('Poro-Perm Cross Plot by Facies')
    plt.show()
else:
    print("'facies' column does not exist in the DataFrame.")

# P-P plots (probability-probability plots)
stats.probplot(dfp['Porosity (%)'], dist="norm", plot=plt)
plt.title('P-P Plot for Porosity')
plt.show()

stats.probplot(dfp['Permeability (mD)'], dist="norm", plot=plt)
plt.title('P-P Plot for Permeability')
plt.show()


# In[12]:


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
X = dfp['Porosity (%)']
Y = dfp['Permeability (mD)']
plt.scatter(X,Y)
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()

X,Y = np.array(X),np.array(Y)
X = X.reshape(-1,1)

model = LinearRegression()
model.fit(X, Y)
r_sq = model.score(X, Y)
print('r_sq = ', r_sq)

y_pred = model.predict(X)

plt.scatter(X,Y)
plt.plot(X,y_pred, color="k")

plt.xlabel('independent')
plt.ylabel('dependent')

plt.show()


# In[13]:


Y = Y.reshape(-1,1)
X = X.reshape(-1,1)
XY = np.hstack((X,Y))
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(XY)

# plot clusters
plt.scatter(
    XY[y_km == 0, 0], XY[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    XY[y_km == 1, 0], XY[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    XY[y_km == 2, 0], XY[y_km == 2, 1],
    s=50, c='red',
    marker='o', edgecolor='black',
    label='cluster 3'
)


# In[14]:


logi1 = dfp['Facies'] == "'overbanks'"
logi1


# In[15]:


POB = dfp['Porosity (%)'][logi1]


# In[16]:


logi2 = dfp['Facies'] == "'channel'"
logi2


# In[17]:


PCB = dfp['Porosity (%)'][logi2]


# In[18]:


logi3 = dfp['Facies'] == "'crevasse splay'"
logi3


# In[19]:


logi = dfp['Facies']=="'overbanks'"
logi1 = dfp['Facies']=="'channel'"
logi2 = dfp['Facies']=="'crevasse splay'"

pOB = dfp['Porosity (%)'][logi]
kOB = dfp['Permeability (mD)'][logi]
pCH = dfp['Porosity (%)'][logi1]
kCH = dfp['Permeability (mD)'][logi1]

plt.scatter(pOB,kOB)
plt.xlabel('independent overbank')
plt.ylabel('dependent overbank')
plt.show()

pOB,kOB = np.array(pOB),np.array(kOB)
pOB = pOB.reshape(-1,1)
model1 = LinearRegression()
model1.fit(pOB, kOB)
r_sq1 = model1.score(pOB, kOB)
print('r_sq = ', r_sq1)

y_pred1 = model1.predict(pOB)
plt.scatter(pOB,kOB)
plt.plot(pOB,y_pred1, color="k")
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()


# In[20]:


pCH = dfp['Porosity (%)'][logi1]
kCH = dfp['Permeability (mD)'][logi1]

plt.scatter(pCH,kCH)
plt.xlabel('independent channel')
plt.ylabel('dependent channel')
plt.show()

pCH,kCH = np.array(pCH),np.array(kCH)
pCH = pCH.reshape(-1,1)
model2 = LinearRegression()
model2.fit(pCH, kCH)
r_sq2 = model2.score(pCH, kCH)
print('r_sq = ', r_sq2)

y_pred2 = model2.predict(pCH)
plt.scatter(pCH,kCH)
plt.plot(pCH,y_pred2, color="k")
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()


# In[21]:


pCS = dfp['Porosity (%)'][logi2]
kCS = dfp['Permeability (mD)'][logi2]

plt.scatter(pCS,kCS)
plt.xlabel('independent crevasse splay')
plt.ylabel('dependent crevasse splay')
plt.show()

pCS,kCS = np.array(pCS),np.array(kCS)
pCS = pCS.reshape(-1,1)
model3 = LinearRegression()
model3.fit(pCS, kCS)
r_sq3 = model3.score(pCS, kCS)
print('r_sq = ', r_sq3)

y_pred3 = model3.predict(pCS)
plt.scatter(pCS,kCS)
plt.plot(pCS,y_pred3, color="k")
plt.xlabel('independent')
plt.ylabel('dependent')
plt.show()


# In[22]:


x = dfp.drop('Facies',axis=1)
y = dfp['Facies']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(x_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[24]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(4,4,4),max_iter=1000)

mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)

import numpy as np
result = np.vstack((y_test, predictions))
print(result)


# In[25]:


logi = predictions == y_test

100 / len(logi) *np.sum(logi)


# In[26]:


plt.hist(dfp['Facies'],25)
plt.xlabel('facies')
plt.ylabel('n')
plt.show()


# In[27]:


# Anderson-Darling Test
from scipy.stats import anderson
# normality test
result_1 = anderson(dfp['Porosity (%)'])
print('Statistic: %.3f' % result_1.statistic)
p = 0
# interpret results
for i in range(len(result_1.critical_values)):
    slevel, cvalues = result_1.significance_level[i], result_1.critical_values[i]
    if result_1.statistic < result_1.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (slevel, cvalues))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (slevel, cvalues))


# In[28]:


# Anderson-Darling Test
from scipy.stats import anderson
# normality test
result_2 = anderson(dfp['Permeability (mD)'])
print('Statistic: %.3f' % result_1.statistic)
p = 0
# interpret results
for i in range(len(result_2.critical_values)):
    slevel, cvalues = result_2.significance_level[i], result_2.critical_values[i]
    if result_2.statistic < result_2.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (slevel, cvalues))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (slevel, cvalues))


# # Image Analysis

# In[29]:


from skimage import io, restoration, measure, feature, color
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the image
image_path = 'berea8bit.tif'
image = io.imread(image_path)

# Apply non-local means denoising
denoised_image = restoration.denoise_nl_means(image, h=0.05)  # Adjust 'h' as needed

# Convert to binary image (adjust threshold as needed)
threshold = np.mean(denoised_image)  # Example threshold
binary_image = denoised_image < threshold

# Simple porosity estimation
porosity = np.sum(binary_image) / np.prod(binary_image.shape)

# Kozeny-Carman permeability estimation
def estimate_permeability(porosity, grain_size):
    return (1/(72*math.pi))*(porosity**3)*(grain_size**2)/((1-porosity)**2)
grain_size=0.1
permeability = estimate_permeability(porosity, grain_size)

# Blob analysis for pore size
blobs = feature.blob_log(binary_image, max_sigma=30, threshold=0.09)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(denoised_image, cmap='gray')
ax[1].set_title('Denoised Image')

ax[2].imshow(binary_image, cmap='gray')
ax[2].set_title('Binary Image')

# Blob visualization
ax[3].imshow(binary_image, cmap='gray')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=True)
    ax[3].add_patch(c)
ax[3].set_title('Blob Analysis')

plt.tight_layout()
plt.show()

# Output results
print(f"Estimated Porosity: {porosity}")
print(f"Estimated Permeability: {permeability}")


# In[ ]:




