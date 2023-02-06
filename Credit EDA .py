#!/usr/bin/env python
# coding: utf-8

# # Analysing the application_data.csv data

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")


# ## Loading data

# In[2]:


df=pd.read_csv("application_data.csv")


# In[3]:


df.head()


# ## Metadata of the data

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.nunique()


# ## Removing Null Value

# In[8]:


# listing the null values columns having more than 40%

df1=df.isnull().sum()
df1=df1[df1.values>(0.4*len(df1))]
len(df1)


# #### So, there are 64 columns having null values greater than 40% in the dataset

# In[9]:


df1 = list(df1[df1.values>=0.4].index)
df.drop(labels=df1,axis=1,inplace=True)
print(len(df1))


# In[10]:


df.isnull().sum()/len(df)*100


# In[11]:


# Filling missing values with median

misvalues=df['AMT_ANNUITY'].median()

df.loc[df['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']=misvalues


# In[12]:


df.isnull().sum()


# In[13]:


df2=df.isnull().sum(axis=1)
df2=list(df2[df2.values>=0.3*len(df)].index)
df.drop(labels=df2,axis=0,inplace=True)
print(len(df2))


# In[14]:


# We will remove unwanted columns from this dataset

unwanted=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df.drop(labels=unwanted,axis=1,inplace=True)


# In[15]:


df.shape


# #### There are some columns where the value is mentioned as 'XNA' which means 'Not Available'. So we have to find the number of rows and columns and implement suitable techniques on them to fill those missing values or to delete them.

# In[16]:


df[df['ORGANIZATION_TYPE']=='XNA'].shape


# In[17]:


df['CODE_GENDER'].value_counts()


# In[18]:



df['ORGANIZATION_TYPE'].describe()


# In[19]:


# Hence, dropping the rows of total 55374 have 'XNA' values in the organization type column

df=df.drop(df.loc[df['ORGANIZATION_TYPE']=='XNA'].index)
df[df['ORGANIZATION_TYPE']=='XNA'].shape


# #### We have DAYS_BIRTH column which can used to derive the age of the customer.

# In[20]:


df["AGE"] = df["DAYS_BIRTH"]//-365.25
bins = [0,20,25,30,35,40,45,50,55,60,100]
slots = ["0-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55","55-60","60 Above"]

df["AGE_GROUP"] = pd.cut(df["AGE"], bins=bins, labels=slots)


# In[21]:


df.AGE.describe()


# In[22]:


df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"]//-365.25
bins = [0,5,10,15,20,25,30,50]
slots = ["0-5","5-10","10-15","15-20","20-25","25-30","30 Above"]

df["EMPLOYEMENT_YEARS"] = pd.cut(df["YEARS_EMPLOYED"], bins=bins, labels=slots)


# In[23]:


df["EMPLOYEMENT_YEARS"].value_counts(normalize= True)*100


# In[24]:


day_col =["DAYS_BIRTH","DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH"]

for i in day_col:
    df[i]=df[i].apply(lambda x : abs(x) if x<0 else x)


# ### Now, Creating bins for continous variable categories column 'AMT_INCOME_TOTAL' and 'AMT_CREDIT'

# In[25]:


df['IncomeRange'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=[0,0.25,0.50,0.90,1], labels=['Low','Average','High','Very High'])


# In[26]:


100*df['IncomeRange'].value_counts(normalize=True)


# In[27]:


df.nunique()


# In[28]:


Cont_cols=["CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED","DAYS_REGISTRATION","AGE","YEARS_EMPLOYED","HOUR_APPR_PROCESS_START","DAYS_ID_PUBLISH"]
Cat_cols=["TARGET","NAME_CONTRACT_TYPE",'CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START',
        'ORGANIZATION_TYPE','FLAG_OWN_REALTY','FLAG_OWN_CAR','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',"IncomeRange","AGE_GROUP","EMPLOYEMENT_YEARS"]
id_cols=["SK_ID_CURR"]


# ## Outliers

# In[29]:


for i in Cont_cols:
    sns.boxplot(y=df[i])
    plt.title("Statistical Distribution of "+i)
    plt.show()


# #### Insight
# #### AMT_ANNUITY, AMT_CREDIT, CNT_CHILDREN, DAYS_EMPLOYED have some number of outliers.
# #### AMT_INCOME_TOTAL has huge number of outliers which indicate that few of the loan applicants have high income when compared to the others.
# #### DAYS_BIRTH has no outliers which means the data available is reliable.
# 

# # Data Analysis

# ## Univarient

# In[30]:


for column in Cat_cols:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    df[column].value_counts().plot(kind='bar',color="green")
    plt.title(column)


# In[31]:


# There are many cilent who apply for cash loan
# There are large number of Female who apply for loan
# There are large number of client who have not own car
# There are large number of client who have own house or flat
# A large number of client apply for loan are of working catogory 
# People with Secondary or secondary special education applied the highest number of loans
# A large number of client who apply for loan are Married
# A large number of client who apply for laon are live in House or apartment
# Most of the people who opted for loan, didn't mention their occupation type. Other high number of people who applied for loans are labourers, Sales Staff and Core staff.
# On Tuesday large number of client apply for loan 
# Age is almost evenly distributed with maximum density between 35-40.
# people have 0-5years employement year
# All the columns REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY which chceks the address given by the applicant is matching in most of the cases here.
# Here are relatively few people with very high income or average who have applied for loans but most of them who have applied for loans either have high income or low income.


# In[32]:


class_values = (df['TARGET'].value_counts()/df['TARGET'].value_counts().sum())*100
print(class_values)


# In[33]:


Non_Defaulters = df[df.TARGET == 0]
Defaulters = df[df.TARGET == 1]
df.head()


# In[34]:


plt.figure(figsize=[8,6]) # Setting the plot size for ease of readabilty
plt.style.use('ggplot')
plt.pie(class_values, labels=['Non_Defaulters (0)', 'Defaulters (1)'],explode=(0,0.05), autopct='%.2f')
plt.title('Defaults VS Non_Defaulters \n', fontsize=20) # Setting the title for the plot
plt.show()


# ## Analysis of Continuous variables of Applicants making payments on time.

# In[35]:


for i in Cont_cols:
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    sns.distplot(Non_Defaulters[i])
    plt.title(i)


# ## Analysis of Continuous variables of Applicants with payment difficulties

# In[36]:


for column in Cont_cols:
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    sns.distplot(Defaulters[column])
    plt.title(column)


# # Bivarient Analysis

# In[37]:


for i in Cat_cols:
    plt.figure(figsize=(30,6))
    plt.subplot(121)
    sns.countplot(df[i],hue=df['TARGET'],data=df)
    plt.title(i)    
    plt.xticks(rotation=90) 


# #### The proportion of people opting out for Cash loans and paying the amount back is more than the people opting for revolving loans.
# #### The proportion of females applying for loans and having difficulties in payment is much more than males
# ####  The number of people working for income are more than any other category. But the number of people having difficulty to pay the more are also from working category people
# ####  Applicants who are Secondary or special secondary educated have applied most of the loans but are also the population facing most of the difficulties while paying the amount
# ####  Married people apply for the most number of loans but tend to have difficulties in the payment as well.
# ####  The number of people live in house/apartment and also they have difficulties in payment as well.
# #### Most of the people apply for loan on Tuesday and they have difficulties in payment as well.
# #### People belonging to organizations like Business entity type 3, unknown organization or are self- employed have applied for the most number of loans respectively and most number of people having propblems in paying back the amount is Business entity type 3, are self- employed or unknown organization respectively.
# #### People in Very High income range tends to repay their loans on time more than defaulting the loans.People in Low income range tends to default their loans more than repaying on time.
# #### People in the age group 35-40 applied the most number of loans and they have difficulties in payment as well.

# In[38]:


sns.barplot(data=df,x='NAME_FAMILY_STATUS',y='AGE',ci=None)
plt.xticks(rotation=90)
plt.title("AGE VS NAME_FAMILY_STATUS")
plt.show()


# #### Mostly client with more age who apply for loan is window

# In[39]:


sns.lineplot(data=df,x="AGE",y="AMT_INCOME_TOTAL")
plt.title("AGE VS AMT_INCOME_TOTAL")

plt.show()


# #### As the Age increase income of the client decrease

# In[40]:


sns.scatterplot(data=df,x='AMT_CREDIT',y='AMT_ANNUITY',hue='TARGET')
plt.show()


# #### An increasing relationship is established between AMT_ANNUITY and AMT_CREDIT.
# #### It becomes difficult for applicant to pay if the 'AMT_CREDIT and AMT_ANNUITY'rises together

# In[41]:



plt.figure(figsize = (18,6))
sns.heatmap(round(df.corr(),3), annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()


# #### The positive correlation between CNT_CHILDREN and CNT_FAM_MEMBERS
# #### The positive correlation between AMT_CREDIT and AMT_ANNUITY

# In[42]:


Cont_cols=df.select_dtypes(include=np.number)
numeric_cols=Cont_cols.columns

Cat_cols=df.select_dtypes(exclude=np.number)
categoric_cols=Cat_cols.columns


# In[43]:


app_T0 = df[df.TARGET == 0]
app_T1 = df[df.TARGET == 1]
df.head()


# In[44]:


app_T0=app_T0.drop(['SK_ID_CURR', 'TARGET'], axis=1)


# In[45]:


app_T1=app_T1.drop(['SK_ID_CURR', 'TARGET'], axis=1)


# In[46]:


corr_app = abs(round(app_T0.corr(),2))
corr_app


# In[47]:


plt.figure(figsize = (18,6))
sns.heatmap(round(app_T0.corr(),3), annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()


# In[48]:


corr_T0 = corr_app[corr_app!=1].unstack().sort_values(ascending = False).head(20)
print("The top 10 correlation pairs for Target = 0 are:")
corr_T0


# In[49]:


corr_app_t1 = abs(round(app_T1.corr(),2))
corr_app_t1


# In[50]:


plt.figure(figsize = (15,6))
sns.heatmap(round(app_T1.corr(),3), annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()


# In[51]:


corr_T1 = corr_app_t1[corr_app_t1!=1].unstack().sort_values(ascending = False).head(20)
print("The top 10 correlation pairs for Target = 1 are:")
corr_T1


# #### For applicants who have difficulties in paying back the amounts, they have very less correlation between there total income(AMT_INCOME_TOTAL) and series of payments made at equal intervals(AMT_ANNUITY)
# #### For defaulters there is a linearly increasing relation, where there applicants permanent address and city doesn’t match with contact address or city to work address or city.
# 

# # Analysing the previous_application.csv data

# In[52]:


dfp=pd.read_csv("previous_application.csv")


# In[53]:


dfp.head()


# # Metadata of the data

# In[54]:


dfp.shape


# In[55]:


dfp.info()


# In[56]:


dfp.describe()


# In[57]:


# listing the null values columns having more than 40%

df2=dfp.isnull().sum()
df2=df2[df2.values>(0.4*len(df2))]
len(df2)


# In[58]:


df2 = list(df2[df2.values>=0.4].index)
dfp.drop(labels=df2,axis=1,inplace=True)
print(len(df2))


# In[59]:


dfp.isnull().sum()/len(dfp)*100


# In[60]:


dfp.shape


# In[61]:


dfp.NAME_CASH_LOAN_PURPOSE.value_counts()


# In[62]:


dfp.CODE_REJECT_REASON.value_counts()


# In[63]:


dfp[dfp['CODE_REJECT_REASON'] == 'XAP']['NAME_CONTRACT_STATUS'].unique()


# In[64]:


dfp[dfp['NAME_CASH_LOAN_PURPOSE'] == 'XNA']['NAME_CONTRACT_TYPE'].unique()


# In[65]:


#Removing null values i.e. rows where NAME_CASH_LOAN_PURPOSE = XNA 
dfp=dfp.drop(dfp[dfp['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)


# In[66]:


dfp.NAME_CASH_LOAN_PURPOSE.value_counts()


# In[67]:


def plot_uni(var):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    
    sns.countplot(x=var, data=dfp,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    plt.show()


# In[68]:


plot_uni('NAME_CONTRACT_TYPE')


# #### From the above chart, we can infer that, most of the applications are for 'Cash loan' and 'Consumer loan'. Although the cash loans are refused more often than others.

# In[69]:


plot_uni('NAME_PAYMENT_TYPE')


# #### From the above chart, we can infer that most of the clients chose to repay the loan using the 'Cash through the bank' option We can also see that 'Non-Cash from your account' & 'Cashless from the account of the employee' options are not at all popular in terms of loan repayment amongst the customers.

# In[70]:


plot_uni('NAME_CLIENT_TYPE')


# #### most of the loan applications are from repeat customers, out of the total applications 70% of customers are repeaters. They also get refused most often.

# # Merging current and previous data sets

# In[71]:


merge_df = df.merge(dfp, on = 'SK_ID_CURR', how = 'inner')
merge_df


# In[72]:


merge_df.shape


# In[73]:


merge_df.head()


# In[74]:


merge_df.nunique()


# In[75]:


# Listing down columns which are not needed
Unnecessary_prev = ['WEEKDAY_APPR_PROCESS_START_y','HOUR_APPR_PROCESS_START_y','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']

merge_df.drop(Unnecessary_prev,axis =1, inplace = True)

merge_df.shape


# In[76]:


merge_df.nunique()


# In[77]:


merge_df=merge_df.drop(merge_df[merge_df['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)


# In[78]:


merge_fi = merge_df.filter(['SK_ID_CURR', 'TARGET','NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AgeGroup',
       'IncomeRange', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS',
       'CODE_REJECT_REASON'], axis = 1)

merge_fi.drop('SK_ID_CURR', axis = 1, inplace = True)

merge_fi.info()


# ### Univariate Analysis

# In[79]:


plt.figure(figsize=(12,4))
plt.subplot(121)
merge_df['NAME_CONTRACT_STATUS'].value_counts().plot(kind='bar', color = 'green')
plt.title('NAME_CONTRACT_STATUS')
plt.show()


# In[80]:


## Maximum number of loans applied by existing customers previously have been refused.


# In[81]:


plt.figure(figsize=(15,6))
plt.subplot(121)
merge_df['NAME_CASH_LOAN_PURPOSE'].value_counts().plot(kind='bar', color = 'green')
plt.title('NAME_CASH_LOAN_PURPOSE')
plt.show()


# In[82]:


#Loans have been requested maximum for Repairs, Urgent needs and unspecified categories. There are almost no loans issued for a third person or because of customer refusal to reveal the objective of taking a loan.


# In[83]:


plt.figure(figsize=(20,20))
plt.xscale('log')
ax=sns.countplot(data = merge_df,y='NAME_CASH_LOAN_PURPOSE', order=merge_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='deep')
plt.title('Distribution of contract status with purposes')
plt.show()


# 
# #### besides (XAP)Most rejection of loans came from purpose 'Repairs'. For education purposes we have equal number of approves and rejection PayinG other loans and buying a new car is having significant higher rejection than approves.

# ### Bivarient Analysis

# In[84]:


merge_app_Refused = merge_fi[merge_fi["NAME_CONTRACT_STATUS"]  == 'Refused']
merge_app_Approved =merge_fi[merge_fi["NAME_CONTRACT_STATUS"]  == 'Approved']
merge_app_Canceled =merge_fi[merge_fi["NAME_CONTRACT_STATUS"]  == 'Canceled']
merge_app_Unused = merge_fi[merge_fi["NAME_CONTRACT_STATUS"]  == 'Unused offer']


# In[85]:


i = 0

for column in merge_app_Unused:
    if column != 'TARGET':
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey = True)
       

        sns.countplot(ax = axes[0], x=merge_app_Unused[column],hue=merge_app_Unused['TARGET'],data=merge_app_Unused)
        axes[0].set_title('Unused Loans')
        
        sns.countplot(ax = axes[1], x=merge_app_Approved[column],hue=merge_app_Approved['TARGET'],data=merge_app_Approved)
        axes[1].set_title('Approved Loans')

        sns.countplot(ax = axes[2], x=merge_app_Refused[column],hue=merge_app_Refused['TARGET'],data=merge_app_Refused)
        axes[2].set_title('Refused Loans')

        sns.countplot(ax = axes[3], x=merge_app_Canceled[column],hue=merge_app_Canceled['TARGET'],data=merge_app_Canceled)
        axes[3].set_title('Canceled Loans')

        for i in range(4):
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(90)


# In[86]:


#In case of refused loans, a HUGE-HUGE proportion of applicants were refused of loans even if they wouldn’t have ended defaulting across all age groups.
#Out of all the loans the bank has approved, the bank has made proportionally more losses for Cash loans compared to revolving loans because there were applicants who couldn't pay their installments on time.
#Out of all the loans the bank has approved, the bank has made proportionally more losses when they have approved loans to males.
#The ratio of Refused loans is higher for defaulting males, although the count of applications shows a higher trend in female population.
#Customers having secondary or secondary special education show highest trend in Refused as well as Approved loans and to a lesser extent in Cancelled loans.
#Highest percentage of defaulters can be observed in case where Cash loans have been requested for Repairs. Highest percentage of Refusal of loans also is in the same category
#Refusal of loans is highest for customers earning a high income. This rate is significantly high for people falling into the age group of 30-40 years.


# In[87]:


plt.figure(figsize=(10,5))
sns.countplot(x=merge_df['NAME_CASH_LOAN_PURPOSE'],hue=merge_df['TARGET'],data=merge_df)
plt.xticks(rotation = 90)
plt.show()


# In[88]:


"NAME_CONTRACT_STATUS",merge_df,"TARGET",['g','r'],False,(14,8)
r = merge_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]
df_new = pd.concat([r.value_counts(),round(r.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
df_new['Percentage'] = df_new['Percentage'].astype(str) +"%" 
df_new


# #### 90% of the previously cancelled client have actually repayed the loan. Revising the interest rates would increase business opportunity for these clients
# #### 88% of the clients who have been previously refused a loan has payed back the loan in current case.
# #### Refusal reason should be recorded for further analysis as these clients could turn into potential repaying customer.

# In[89]:



plt.figure(figsize = (18,6))
sns.heatmap(round(merge_df.corr(),3), annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()


# #### observe a correlation trend similar to the one we observed while analysing application data.

# # Conclusion

# ## The bank has lost quite a good amount of profits by not approving revolving loans to applicants who would have paid the installments on time. The bank needs to increase their revolving loans number so as to derive continuous profits from the same depending on the amount of cash reserve the bank has.
# ## Males with high income and secondary education are more likely to have difficulties while paying back the amount or default. Hence while approving their loans, more attention to detail needs to be given.
# ## Maximum number of loans applied by existing customers previously were refused. So now when they have applied again, if total income has imcreased or credit amount has decreased or the annuity has decreased such that now the payment is possible- then loans can be approved to them. Also if now the target variable for those applicants shows 0- loans can be approved for them.
# ## Loan taken for the purpose of Repairs seems to have highest default rate. A very high number applications have been rejected by bank or refused by client in previous applications as well which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected, or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan. The same approach could be followed in future as well.

# In[ ]:




