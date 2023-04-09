#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import KMeans,AgglomerativeClustering
from kneed import KneeLocator ## for elbow
from sklearn.metrics import silhouette_score,calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer,silhouette_visualizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import warnings
import geopandas as gpd
import folium
from mpl_toolkits.axes_grid1 import make_axes_locatable
warnings.simplefilter("ignore")


# In[2]:


data=pd.read_csv('D:/ML/imt/ML/DataSet/Country-data_1.csv',encoding='latin-1')


# In[3]:


data


# ### Data Dictionary :
# child morth=Death of children under 5 years of age per 1000 live births
# 
# exports=Exports of goods and services per capita. Given as % age of the GPD per capita
# 
# health=Total health spending per capita Given as % age of the GPD per capita
# 
# imports=Imports of goods and services per capita. Given as % age of the GPD per capita
# 
# income=Net income per person
# 
# inflation= The measurment of the annual growth rate of the Total GPD
# 
# life_expec=The average number of years a new born child would live if the current mortality patterns are to remain the same.
# 
# Total_fer=The number of children that would be born to each woman if the current age-fertility rates remain the same.
# 
# gdpp=The GDP per capita. Calculated as the Total GPD divided by the total population.
# 

# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# ##### None of the columns have null values hence no imputation or drop required.

# #### Convert % columns to actual values:

# In[6]:


data['exports'] =data['exports'] * data['gdpp']/100
data['imports'] = data['imports'] * data['gdpp']/100
data['health'] = data['health'] * data['gdpp']/100


# #### I merged the data with the Geographical coordinates of countries and made a new data frame.

# In[7]:


cdata=pd.read_csv('D:/ML/imt/ML/DataSet/country_1.csv')
cdata


# In[8]:


df= pd.merge(data, cdata,how='left',on=['country'])
df.pop('Unnamed: 4')


# # Data Analysis

# In[9]:


print(f"This Dataset Contains Information about {data['country'].nunique()} Countries that needs financial support!")


# In[10]:


fig,ax=plt.subplots(figsize=(10,5))
sns.heatmap(data.corr(),cmap='RdBu_r',cbar=True,annot=True,linewidths=0.5,ax=ax)
plt.show()


# ##### 1.Child mortality is highly correlated with total fertility  and inverse correlated with life expectancy.
# ##### 2.imports and exports are highly correlated.
# ##### 3.Income and gdpp are highly correlated.
# ##### 4.Health and gdpp are highly correlated.

# In[11]:


plt.figure(figsize=(50,10))
sns.barplot(x='country',y='child_mort',data=data.sort_values(by='child_mort',ascending=False))
plt.title('   Child mortality per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.grid()
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('Country',fontsize=20)
plt.ylabel('Child mortality',fontsize=20)
plt.savefig('D:/ML/imt/pic_pca/child_mort.jpg')
plt.show()


# #### As show above Haiti, Sierra Leone, Central African, Mali,Nigeria are the countries with the most child mortality.

# #### Show countries on the map.
# To show countries on the map, I used geopandas package so I needed to merge the data with the geopandas dataset.

# In[12]:


path_to_data = gpd.datasets.get_path("naturalearth_lowres")
gdf = gpd.read_file(path_to_data)


# In[13]:


data_world=gdf.merge(data,how='left',left_on=['name'], right_on=['country'])


# In[14]:


data_world.plot(figsize=(20,16))
x = df['longitude']
y = df['latitude']
z = df['child_mort']
plt.scatter(x, y, s=z*2, c='red', alpha=0.6, vmin=0, vmax=10)
plt.title("Child mortality in Countries")
plt.show()


# #### The size of the points shows the amount of child mortality. As you see above, most child mortalities are in Africa continent.

# In[15]:


ax = data_world["geometry"].plot(figsize=(20,16))
data_world.plot( column="total_fer", ax=ax, cmap='YlOrRd', legend=True, legend_kwds={"label": "Total fertility", "orientation":"horizontal"},missing_kwds={"color": "white", "edgecolor": "white", "hatch": "|"})
ax.set_title("Total fertility in Countries")
plt.show()
plt.show()


# #### Red areas show more fertility in women.

# In[16]:


sns.boxplot(x='continent',y='life_expec',data=data_world)
sns.stripplot(x='continent',y='life_expec',data=data_world)
plt.xticks(rotation=45,fontsize=10)
plt.title('Life Expectancy per continent')
plt.show()


# #### Evidently, African life spans are substantially shorter than those in other regions of the world, with a typical expectancy of less than seventy-five years.

# In[17]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.swarmplot(x='continent',y='income',data=data_world)
plt.xticks(rotation=45,fontsize=10)
plt.title('Income per continent')
plt.subplot(1,2,2)
sns.swarmplot(x='continent',y='gdpp',data=data_world)
plt.xticks(rotation=45,fontsize=10)
plt.title('gdpp per continent')
plt.show()


# #### Most countries in Africa, Asia, and South America experience a lower Gross Domestic Product (GDP) and income than those in other regions.

# In[18]:


ax = data_world["geometry"].plot(figsize=(20,16))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad="0.5%")
data_world.plot( column="health", ax=ax, cax=cax, cmap='Greens', legend=True, legend_kwds={"label": "Health"},missing_kwds={"color": "lightgrey"})
ax.set_title("Total health spending per capita in Countries")
plt.show()


# #### North America and Europe have higher health expenditures per individual than other areas.

# In[19]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.histplot(x='continent',y='exports',data=data_world,hue='continent')
plt.xticks(rotation=45,fontsize=10)
plt.title('Export per continent')
plt.subplot(1,2,2)
sns.histplot(x='continent',y='imports',data=data_world,hue='continent')
plt.xticks(rotation=45,fontsize=10)
plt.title('Import per continent')
plt.show()


# In[20]:


data_world.plot(figsize=(20,16))
x = df['longitude']
y = df['latitude']
z = df['inflation']
plt.scatter(x, y, s=z*5, c='red', alpha=0.6, vmin=0, vmax=10)
plt.title("inflation in Countries")
plt.annotate('Nigeria',
            xy=(8.5,6),  # theta, radius
            xytext=(0.4,0.4),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right',
            verticalalignment='bottom',
            )
plt.annotate('Nigeria',
            xy=(8.5,6),  # theta, radius
            xytext=(0.4,0.4),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right',
            verticalalignment='bottom',
            )
plt.show()


# #### Nigeria is having high inflation rate

# ## Scaled Data
# Here, we will use Min-Max scaling.

# In[21]:


country=data.pop('country')


# In[22]:


scaler=MinMaxScaler(feature_range=(0,1))
norm1=scaler.fit_transform(data)
norm_df=pd.DataFrame(norm1,columns=data.columns)


# ## PCA

# In[23]:


pca=PCA(n_components=3,svd_solver='auto',random_state=42)


# In[24]:


pca.fit(norm_df)
pca_data=pca.transform(norm_df)


# In[25]:


ax = plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.title('Variance Ratio bar plot for each PCA components.')
plt.xlabel("PCA Components",fontweight = 'bold')
plt.ylabel("Variance Ratio",fontweight = 'bold')
plt.show()


# Around 90% of variance is explained by the Fisrt3 columns
# 

# In[26]:


fig, ax = plt.subplots()
xi = np.arange(1,4, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.89, color='r', linestyle='-')
plt.text(1, 0.9, '89% cut-off threshold', color = 'red', fontsize=16)
plt.show()


# In this case, for 3 principal components get 89% of variance .

# In[27]:


# Checking which attributes are well explained by the pca components
colnames = list(data.columns)
pca_attr = pd.DataFrame({'Attribute':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2]})
pca_attr


# child_mort,total_fer are very well explained by PC2
# 
# life_expec is explained by PC1
# 
# income,export,inflation is explained by PC3
# 
# gdpp,health are well explained by the components PC1,PC2
# 

# ## Train Model

# In[28]:


pca_final = pd.DataFrame(pca_data, columns=["PC1", "PC2","PC3"])
pca_final


# ### Finding the Optimal Number of Clusters
# 
# ###### Elbow Curve to get the right number of Clusters¶
# 

# In[29]:


model = KMeans(random_state=42)
distortion_visualizer = KElbowVisualizer(model, k=(2,8))

distortion_visualizer.fit(pca_final)       
distortion_visualizer.show()    
print("Looking at the above elbow curve it looks good to proceed with either", distortion_visualizer.elbow_value_,"clusters.")


# #### Silhouette Analysis
# 

# In[30]:


kmeans_silhouette_score=[]
for k in range(2,8):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(pca_final)
    score=silhouette_score(pca_final,kmeans.labels_, metric='euclidean')
    kmeans_silhouette_score.append(score)

plt.style.use('fivethirtyeight')
plt.plot(range(2,8),kmeans_silhouette_score)
plt.xticks(range(2,8))
plt.xlabel('Number of clusters')
plt.ylabel('silhouette Score')
#plt.axvline(x=k1.elbow,color='r',label='axvline - full height',ls='--')
plt.show()


# #### Calinski harabasz score

# In[31]:


kmeans_calinski_score=[]
for k in range(2,8):
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_final)
    score1=calinski_harabasz_score(pca_final,kmeans.labels_ )
    kmeans_calinski_score.append(score1)

plt.style.use('fivethirtyeight')
plt.plot(range(2,8),kmeans_calinski_score)
plt.xticks(range(2,8))
plt.xlabel('Number of clusters')
plt.ylabel('CH Score')
#plt.axvline(x=k1.elbow,color='r',label='axvline - full height',ls='--')
plt.show()


# ##### Resualt: From Calinski harabasz score, Silhouette's score,we can see that we have the maximum at 3 and According to elbow hint  we have a breakpoint at 3,so cluster 3 is the best number of clusters.

# In[32]:


df_pca_final = pd.concat([country, pca_final], axis=1)


# In[33]:


kmeans=KMeans(n_clusters=3,random_state=42)
cl=kmeans.fit(pca_final)


# In[34]:


df_pca_final['Cluster']=cl.labels_


# In[35]:


df_pca_final


# In[36]:


fig, axes = plt.subplots(1,3, figsize=(20,5))
sns.scatterplot(data=df_pca_final, x="PC1", y="PC2", hue = "Cluster" ,ax=axes[0])
plt.xlabel("Principal Component 1",fontweight = 'bold')
plt.ylabel("Principal Component 2",fontweight = 'bold')
sns.scatterplot(data=df_pca_final, x="PC1", y="PC3", hue = "Cluster" ,ax=axes[1])
plt.xlabel("Principal Component 1",fontweight = 'bold')
plt.ylabel("Principal Component 3",fontweight = 'bold')
sns.scatterplot(data=df_pca_final, x="PC2", y="PC3", hue = "Cluster" ,ax=axes[2])
plt.xlabel("Principal Component 2",fontweight = 'bold')
plt.ylabel("Principal Component 3",fontweight = 'bold')
plt.show()


# ### let's now visualize the data on the original attributes

# In[37]:


kmeans=KMeans(n_clusters=3,random_state=42)
cl2=kmeans.fit(data)


# In[38]:


data['country']=country


# In[39]:


df['Cluster']=cl2.labels_
data['Cluster']=cl2.labels_
data_world=gdf.merge(data,how='left',left_on=['name'], right_on=['country'])


# In[40]:


ax=sns.histplot(data=data,x="Cluster")
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()+0.1))
plt.title('Number of countries in each cluster')
plt.show()


# In[41]:


ax = data_world["geometry"].plot(figsize=(20,16))

data_world.plot( column="Cluster", ax=ax, categorical=True, legend=True,
                legend_kwds={'loc': 'center left'},missing_kwds={"color": "white"})
x = df['longitude']
y = df['latitude']
z = df['Cluster']
sns.scatterplot(x, y, hue=z,style=z,size=df['Cluster'],palette='icefire')
ax.set_title("Distribution of countries according to cluster labels")
plt.show()


# In[42]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(1,3,1)
sns.boxplot(x='Cluster',y='income',data=data)
plt.title('Clusters of income')
plt.subplot(1,3,2)
sns.boxplot(x='Cluster',y='gdpp',data=data)
plt.title('Clusters of gdpp')
plt.subplot(1,3,3)
sns.boxplot(x='Cluster',y='inflation',data=data)
plt.title('Clusters of inflation')

plt.show()


# ### According to these plot, countries divided to 3 clusters:
# #### cluster0: Those countries that have low income,gdpp . Income between0-30000 per person and gdpp between0-20000 per person. 
# #### cluster1: Countries with high income. Income between 70000-90000  per person. gdpp between 39000-100000 per person.
# #### cluster2:countries with middle income. Income between20500-60000  per person. gdpp between19000-80000 per person

# In[43]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(2,3,1)
sns.swarmplot(x='Cluster',y='child_mort',data=data)
plt.title('Clusters of Child mortality ')
plt.subplot(2,3,2)
sns.swarmplot(x='Cluster',y='life_expec',data=data)
plt.title('Clusters of Life Expectancy')
plt.subplot(2,3,3)
sns.swarmplot(x='Cluster',y='total_fer',data=data)
plt.title('Clusters of Total fertility')
plt.show()


# #### Cluster 0 demonstrates a heightened level of fertility and child mortality.
# ####  Countries in clusters 1 and 2  have higher life expectancies.
# ### As a result, economically disadvantaged countries are placed in cluster 0.

# It is not the end. I cluster the data in the weakest cluster (here is cluster 0) twice until we arrive at fewer destitute nations.

# In[44]:


data2=data[data['Cluster']==0]
country=data2.pop('country')


# In[45]:


kmeans=KMeans(n_clusters=4,random_state=42)
cl3=kmeans.fit(data2)


# In[46]:


data2['country']=country
data2['Cluster2']=cl3.labels_
df2=df[df['Cluster']==0]
df2['Cluster2']=cl3.labels_
data_world2=gdf.merge(data2,how='left',left_on=['name'], right_on=['country'])


# In[47]:


ax=sns.histplot(data=data2,x="Cluster2")
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()+0.1))
plt.title('Number of countries in each cluster')
plt.show()


# In[48]:


fig, axes = plt.subplots(1,2, figsize=(20,6))
plt.subplot(1,2,1)
sns.boxplot(x='Cluster2',y='income',data=data2)
plt.title('Clusters of income')
plt.subplot(1,2,2)
sns.boxplot(x='Cluster2',y='gdpp',data=data2)
plt.title('Clusters of gdpp')

plt.show()


# In[49]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(2,3,1)
sns.swarmplot(x='Cluster2',y='child_mort',data=data2)
plt.title('Clusters of Child mortality ')
plt.subplot(2,3,2)
sns.swarmplot(x='Cluster2',y='life_expec',data=data2)
plt.title('Clusters of Life Expectancy')
plt.subplot(2,3,3)
sns.swarmplot(x='Cluster2',y='total_fer',data=data2)
plt.title('Clusters of Total fertility')
plt.show()


# In[50]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(2,3,1)
sns.scatterplot(x='income',y='child_mort', hue = "Cluster2" ,data=data2)
plt.title('Clusters of Child mortality and income')
plt.subplot(2,3,2)
sns.scatterplot(x='gdpp',y='child_mort', hue = "Cluster2" ,data=data2)
plt.title('Clusters of child mortality and GDP')
plt.subplot(2,3,3)
sns.scatterplot(x='health',y='child_mort', hue = "Cluster2",data=data2)
plt.title('Clusters of Child mortality and health')
plt.show()


# #### Countries in cluster 1 have a lower income and GDP and have a heightened child death rate and fertility. As a consequence, financially destitute countries are paced in cluster 1.
# 
# As I mentioned above, I cluster the data in the weakest cluster once more(here is cluster 1).

# In[51]:


data3=data2[data2['Cluster2']==1]
country=data3.pop('country')
kmeans=KMeans(n_clusters=4,random_state=42)
cl4=kmeans.fit(data3)


# In[52]:


data3['country']=country
data3['Cluster3']=cl4.labels_
df3=df2[df2['Cluster2']==1]
df3['Cluster3']=cl4.labels_
data_world3=gdf.merge(data3,how='left',left_on=['name'], right_on=['country'])


# In[53]:


ax=sns.histplot(data=data3,x="Cluster3")
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.1))
plt.title('Number of countries in each cluster')
plt.grid()
plt.show()


# In[54]:


fig, axes = plt.subplots(1,2, figsize=(20,6))
plt.subplot(1,2,1)
sns.boxplot(x='Cluster3',y='income',data=data3)
plt.title('Clusters of income')
plt.subplot(1,2,2)
sns.boxplot(x='Cluster3',y='gdpp',data=data3)
plt.title('Clusters of gdpp')

plt.show()


# In[57]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(2,3,1)
sns.swarmplot(x='Cluster3',y='child_mort',data=data3)
plt.title('Clusters of Child mortality ')
plt.subplot(2,3,2)
sns.swarmplot(x='Cluster3',y='life_expec',data=data3)
plt.title('Clusters of Life Expectancy')
plt.subplot(2,3,3)
sns.swarmplot(x='Cluster3',y='total_fer',data=data3)
plt.title('Clusters of Total fertility')
plt.show()


# ##### As above plots show, cluster 1 economically disadvantaged countries are placed in cluster 0

# In[58]:


ax = data_world["geometry"].plot(figsize=(20,16))
x = df3[df3['Cluster3']==0]['longitude']
y = df3[df3['Cluster3']==0]['latitude']
plt.scatter(x, y,c='red')
ax.set_title("Distribution of countries according to cluster labels")
plt.show()


# In[59]:


fig, axes = plt.subplots(1,3, figsize=(20,6))
plt.subplot(2,3,1)
sns.scatterplot(x='income',y='child_mort', hue = "Cluster3" ,data=data3)
plt.title('Clusters of Child mortality and income')
plt.subplot(2,3,2)
sns.scatterplot(x='gdpp',y='child_mort', hue = "Cluster3" ,data=data3)
plt.title('Clusters of child mortality and GDP')
plt.subplot(2,3,3)
sns.scatterplot(x='health',y='child_mort', hue = "Cluster3",data=data3)
plt.title('Clusters of Child mortality and health')
plt.show()


# In[60]:



plt.figure(figsize=(60,30))
plt.subplot(3,3,1)
sns.lineplot(x='country',y='total_fer',data=data3[data3['Cluster3']==0].sort_values(by='total_fer',ascending=False))
plt.title('   Total fertility per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Total fertility',fontsize=30)
plt.subplot(3,3,2)
sns.lineplot(x='country',y='life_expec',data=data3[data3['Cluster3']==0].sort_values(by='life_expec',ascending=False))
plt.title('  Life Expectancy per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Life Expectancy',fontsize=30)
plt.subplot(3,3,3)
sns.lineplot(x='country',y='imports',data=data3[data3['Cluster3']==0].sort_values(by='imports',ascending=False))
plt.title('   Import per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Import',fontsize=30)

plt.subplot(3,3,4)
sns.lineplot(x='country',y='exports',data=data3[data3['Cluster3']==0].sort_values(by='exports',ascending=False))
plt.title('   Export per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Export',fontsize=30)
plt.subplot(3,3,5)
sns.lineplot(x='country',y='gdpp',data=data3[data3['Cluster3']==0].sort_values(by='gdpp',ascending=False))
plt.title('  Gdpp per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('gdpp',fontsize=30)
plt.subplot(3,3,6)
sns.lineplot(x='country',y='income',data=data3[data3['Cluster3']==0].sort_values(by='income',ascending=False))
plt.title('   Income per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Income',fontsize=30)

plt.subplot(3,3,7)
sns.lineplot(x='country',y='inflation',data=data3[data3['Cluster3']==0].sort_values(by='inflation',ascending=False))
plt.title('   Inflation per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Inflation',fontsize=30)
plt.subplot(3,3,8)
sns.lineplot(x='country',y='health',data=data3[data3['Cluster3']==0].sort_values(by='health',ascending=False))
plt.title('  Health per country   ',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('Health',fontsize=30)
plt.subplot(3,3,9)
sns.lineplot(x='country',y='child_mort',data=data3[data3['Cluster3']==0].sort_values(by='child_mort',ascending=False))
plt.title('   Child mortality per country   ',horizontalalignment='center',verticalalignment='baseline',fontsize=30,backgroundcolor='black',c='white')
plt.xticks(rotation=90,fontsize=30)
plt.ylabel('child_mort',fontsize=30)
plt.show()


# In[61]:


aid_need_countries=list(data3[data3['Cluster3']==0].sort_values(by=['child_mort','income','gdpp'],ascending=True).country)


# In[62]:


print('Countries that need of aid : ','\n', aid_need_countries)


# In[ ]:




