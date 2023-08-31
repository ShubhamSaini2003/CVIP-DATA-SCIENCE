#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




# In[ ]:




The Dataset
The Dataset was extracted from the Global Terrorism Database (GTD) - an open-source database including information on terrorist attacks around the world from 1970 through 2017. The GTD includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period and now includes more than 180,000 attacks. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland.

Explanation of selected columns:
success - Success of a terrorist strike
suicide - 1 = "Yes" The incident was a suicide attack. 0 = "No" There is no indication that the incident was a suicide
attacktype1 - The general method of attack
attacktype1_txt - The general method of attack and broad class of tactics used.
targtype1_txt - The general type of target/victim
targsubtype1_txt - The more specific target category
target1 - The specific person, building, installation that was targeted and/or victimized
natlty1_txt - The nationality of the target that was attacked
gname - The name of the group that carried out the attack
gsubname - Additional details about group that carried out the attack like fractions
nperps - The total number of terrorists participating in the incident
weaptype1_txt - General type of weapon used in the incident
weapsubtype1_txt - More specific value for most of the Weapon Types
nkill - The number of total confirmed fatalities for the incident
nkillus - The number of U.S. citizens who died as a result of the incident
# In[56]:


gtd = pd.read_csv(r"C:\Users\iamsh\Downloads\globalterrorismdb_0718dist.csv",encoding=('ISO-8859-1'),low_memory =False)

gtd.head(4)


# In[57]:


#select the columns we will use in this analysis
gtd_df = gtd[['eventid', 'iyear','success','imonth', 'iday', 'country_txt',
              'region_txt','suicide', 'attacktype1_txt', 'targtype1_txt', 'target1','nkill']]


# In[58]:


gtd_df.head(4)


# In[59]:


pd.set_option('display.max_rows', None)


# # Data Cleaning and wrangling

# In[60]:


gtd_df.info()


# In[61]:


gtd_df.isnull().sum()


# In[ ]:





# In[ ]:





# In[62]:


gtd_df.isnull().sum()


# # Explanatory Data Analysis
# Let's first look at the trends over the years for the world.
# 
# 

# In[63]:


yearly_attacks = gtd_df.groupby('iyear').size().reset_index(name='count')


# In[64]:


sns.lineplot(x='iyear', y='count', data=yearly_attacks, color= "red")
plt.xlabel('Years')
plt.ylabel('Number of Attacks')
plt.title('World Wide Terrorism Number of Attack Trends from 1970 to 2017.')
plt.show()


# Here we can see that there has been a fluctuation in the number of terrorism attacks from 1970 to 2004. While during 2005, there was a dramatic increase in terrorism attacks around the globe. Figures were fives time higher by the end of 2017! Let's look into what might have caused that.

# # Terrorist Attack trends based on the region
# 

# In[65]:


yearly_attacks_region = gtd_df.groupby(['iyear', 'region_txt']).size().reset_index(name='count')


# In[66]:


sns.lineplot(x='iyear', y='count',hue='region_txt', data=yearly_attacks_region )
plt.title('Terrorist Attacks Trends in Regions from 1970 to 2017')
plt.xlabel('Years')
plt.ylabel('Number of Attacks')
plt.show()


# Here we can see during the 1970s to the 1980s, most of the terrorist attacks took place in Central Asia, while the figure was relatively low compared to present day, it was still huge when compared to the other regions. During the 1980s to early parts of the 1990s, South America has seen more terrorist attacks, then during the 2000s to 2010s, the number of terrorist attacks in the Middle East and North Africa and South America increased significantly.

# # Countries with the most terrorist attacks
# 

# In[67]:


country_attacks = gtd_df.groupby('country_txt').size().reset_index(name='count').sort_values(by='count', ascending = False)
top5_country = country_attacks.head(5)
sns.barplot(x='country_txt', y='count', data=top5_country)
plt.title('Top 5 countries with the most terrorist attacks from 1970 to 2017')
plt.xlabel('Countries')
plt.ylabel('Number of Terrorist Attacks')
plt.show()


# Looks do a deep dive into the the last 5 years, especially with the dramatic increase of terrorist attacks. What could have possibly contributed to that?

# In[68]:


narrow_2017 = gtd_df[gtd_df['iyear'] >= 2012]
narrow_2017_count = narrow_2017.groupby(['iyear', 'country_txt']).size().reset_index(name='count')

# Get the top 5 countries for each year
narrow_2017_count_5 = narrow_2017_count.groupby('iyear').apply(lambda x: x.nlargest(5, 'count')).reset_index(drop=True)


# In[69]:


sns.lineplot(x='iyear', y='count', hue='country_txt', data=narrow_2017_count_5)


# Here we can see some of the countries with the most terrorist attacks from 2012 to 2017. Iraq had the most terrorist attacks, with the most attacks during 2014.
# 
# 

# In[70]:


gtd_df.head(1)


# # World Wide Outcome of a terrorist attack
# Let's look at the success of the terrorist attacks over the years.

# In[71]:


total_count = gtd_df['success'].count()
success = gtd_df.groupby('success').size().reset_index(name="count")
success['percentage'] = (success['count'] / total_count) * 100


# In[72]:


success


# In[73]:


sns.barplot(x='success', y= 'percentage',data = success)
plt.title('Success rate of Terrorist attacks from 2012 to 2017')


# Here we can see that 89% of incidents are successful! Let's take a deeper dive into each incident, and their number of kills. Could there be a correlation between the incident type and the number of kills?

# In[74]:


attack_type = gtd_df.groupby(['attacktype1_txt', 'success']).size().reset_index(name="count")
attack_type


# In[75]:


plt.figure(figsize=(27,12))
plt.title('Number of each attack type and their success rate from 2012 to 2017')
sns.barplot(x='attacktype1_txt', y='count', hue='success', data=attack_type, color= "red")
plt.xlabel('Attack Type')
plt.ylabel('Number of Attacks')


# Bombing/Explosion had the highest number of terrorist attacks and the highesr success rate!Armed Asssault also had a fairly high number of attacks and a high success rate, arounf half the figure bombing/explosion. Assisnation had quite a higher failure rate than amrmed assault even though it had way less the number of attacks when compared.

# # Casualties due to Terrorist Attacks around the world

# In[76]:


nkills_attack = gtd_df.groupby('attacktype1_txt')[['nkill']].sum().reset_index()
nkills_attack


# In[77]:


plt.figure(figsize=(25,10))
sns.barplot(x='attacktype1_txt', y='nkill', data=nkills_attack)
plt.xlabel('Attack Type')
plt.ylabel('Total Number of Kills')
plt.title('Total Number of Kills by Attack Type')
plt.xticks(rotation=90)
plt.show()


# Despite Armed Assault having way less events and a much lower success rate, it stands as one of the attack types with the highest number of casualties along side bombing and explosion.

# # Let's take a look at my two dearest countries, India and Mexico.

# In[78]:


mycountries = gtd_df[(gtd_df['country_txt'] == 'India') | (gtd_df['country_txt'] == 'Mexico')]
mycountries.head(5)


# # Let's start with India, let's see if we can identify any trend over time.

# # Terrorist Attacks in India

# In[79]:


India = gtd_df[(gtd_df['country_txt'] =='India')]
num_ofattacks = India.groupby('iyear').size().reset_index(name="count")
num_ofattacks


# In[80]:


Ind_sumattacks = num_ofattacks['count'].sum()
Ind_sumattacks

print('Total Number of attacks in India',Ind_sumattacks)


# India had a total of 11960 terrorist attacks.

# In[81]:


plt.figure(figsize=(10,5))
plt.title("Yearly Trend of Terrorist Attcks in India from 1974 t0 2017")
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
sns.lineplot(x="iyear", y="count", data=num_ofattacks)


# # Let's look at the success rate of the attacks over the years.

# In[82]:


succ_Ind = India.groupby(['success']).size().reset_index(name='count')
succ_Ind['percentage'] = succ_Ind['count']/ Ind_sumattacks *100
succ_Ind


# In[83]:


sns.barplot(x = 'success', y = 'percentage', data=succ_Ind)
plt.title("Outcome of Terrorist Attacks in India")
plt.xlabel("Outcome")


# Of all the 11960  attacks 85% were successful, while 15% was unsuccessful.

# # Terrorist Attack Types in India

# In[84]:


India_atype= India.groupby(['attacktype1_txt','success']).size().reset_index(name='count')
India_atype


# In[85]:


plt.figure(figsize=(25,10))
sns.barplot(x='attacktype1_txt', y='count', hue= 'success', data=India_atype, color="red")
plt.title("Terrorist Attacks and their outcome in India")
plt.xlabel("Attack Type")


# here we can see that Bombong/Explosion Attacks was the most frequent type of attack, next to Armed Assualt. Bombing/Explosion has a very high success rate, with only 2500/3500 attacks succesful!

# # Attacks Types in India and Casualties

# In[86]:


nkillattack_Ind = India.groupby('attacktype1_txt')[['nkill']].sum().reset_index()
nkillattack_Ind


# In[87]:


plt.figure(figsize=(25,10))
sns.barplot(x='attacktype1_txt', y='nkill', data=nkillattack_Ind)
plt.title("Attack types in India and their casualties")
plt.xlabel("Attack type")


# here we can see that Armed Assault had the highest number of casualtie of India. Next with Assasination and Bombing/Explosion. All of the others, even though they had some success, thankfull resulted in no casualties.

# # Terrorist Attacks in Mexico.

# Let's take a look at Mexico

# In[88]:


Mexico = gtd_df[(gtd_df['country_txt'] == 'Mexico')]
Mexico.head(1)


# In[89]:


#how many number of attacks were there in Mexico
Mex_attacks = Mexico['eventid'].count()
print('There were',Mex_attacks ,'attacks in Mexico.')


# In[90]:


Mex_success = Mexico.groupby('success').size().reset_index(name='count')
Mex_success['percentage'] = Mex_success['count'] / Mex_attacks * 100
Mex_success


# In[91]:


sns.barplot(x='success', y='percentage', data = Mex_success)
plt.title("Outcome of Terrorist Attacks in Mexico")
plt.xlabel("Outcome")


#  out Of the  524 attacks in Mexico, 91% were successful, while 8% was unsuccessful.

# # Attack types in Mexico and their success rates.

# In[92]:


attack_types_Mexico = Mexico.groupby(['attacktype1_txt','success']).size().reset_index(name='count')
attack_types_Mexico


# In[93]:


plt.figure(figsize=(25,10))
sns.barplot(x='attacktype1_txt', y='count', hue='success', data=attack_types_Mexico, color = "red")
plt.title("Facility ")


# Compared to the world terrorist attacks trend with India, Mexico's highest terrorist attacks comes from Armed Assault.

# In[94]:


#number of kills
nkills_Mexico = Mexico.groupby('attacktype1_txt')[['nkill']].sum().reset_index()
nkills_Mexico


# In[95]:


plt.figure(figsize=(25,10))
sns.barplot(x='attacktype1_txt', y='nkill', data=nkills_Mexico)


# Thankfully it had a much lower number of casuality. Armed Assault and Bombing/Explosion has the highest number of casualties.

# # Conclusion
# 

# 
# Terrorism Attacks all around the world is becoming increasingly a problem! the number of terrorist attacks in the Middle East and North Africa and South America increased significantly. 89% of attacks have been successful, with armed asssault being the most used terrorist attacks and armed assault and bombing/ explosion causing the most casualties.

# In[ ]:




