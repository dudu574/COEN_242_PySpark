#Start a new Spark Session
from pyspark.sql import SparkSession

# App named 'Cruise'
spark = SparkSession.builder.appName('cruise').getOrCreate()


# In[6]:


#Read the csv file in a dataframe
df = spark.read.csv('transactions.csv',inferSchema=True,header=True)


# In[7]:


#Check the structure of schema
df.printSchema()


# In[8]:


df.show()


# In[9]:


df.describe().show()


# In[10]:


# df.groupBy('Cruise_line').count().show()


# In[23]:


#Convert string categorical values to integer categorical values
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="creditLimit", outputCol = "credit_limit_output")
indexed = indexer.fit(df).transform(df)
indexed.head(5)


# In[24]:


from pyspark.ml.linalg import Vectors


# In[25]:


from pyspark.ml.feature import VectorAssembler


# In[26]:


indexed.columns


# In[28]:


# Create assembler object to include only relevant columns 
assembler = VectorAssembler(
inputCols=["creditLimit", "availableMoney", "transactionAmount"],
outputCol="Features")


# In[29]:


output = assembler.transform(indexed)


# In[30]:


output.select("features","creditLimit").show()


# In[31]:


final_data = output.select("features","creditLimit")


# In[32]:


#Split the train and test data into 70/30 ratio
train_data,test_data = final_data.randomSplit([0.7,0.3])


# In[33]:


from pyspark.ml.regression import LinearRegression


# In[34]:


#Training the linear model
lr = LinearRegression(labelCol='creditLimit')


# In[35]:



lrModel = lr.fit(train_data)


# In[39]:



print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# In[40]:


#Evaluate the results with the test data
test_results = lrModel.evaluate(test_data)


# In[41]:


print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))


# In[42]:


from pyspark.sql.functions import corr


# In[43]:


#Checking for correlations to explain high R2 values
df.select(corr('creditLimit','availableMoney')).show()


# In[44]:


df.select(corr('creditLimit','transactionAmount')).show()

