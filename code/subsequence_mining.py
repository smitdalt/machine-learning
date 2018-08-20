'''
sudo vim conf/spark-defaults.conf
#uncomment the spark.driver.memory and change it according to your use. I changed it to below
spark.driver.memory 15g
# press : and then wq! to exit vim editor


export SPARK_HOME=/Users/jayurbain/Dropbox/spark-1.6.2-bin-hadoop2.6
export PYSPARK_PYTHON=/Applications/anaconda/bin/python2.7
#bin/pyspark -Dspark.executor.memory=8g


export PYSPARK_DRIVER_PYTHON=ipython2.7
export IPYTHON=1
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

bin/pyspark


'''
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.types import *
from datetime import datetime
import time
from pyspark.sql.functions import *
from pyspark.mllib.fpm import FPGrowth, PrefixSpan

# sc is an existing SparkContext.
sqlContext = HiveContext(sc)

# load i2b2 data
data = sc.textFile("/Users/jayurbain/Dropbox/machine-learning/machine-learning/data/sample_fpgrowth.txt")

print data.take(10)

# fpgrowth example
transactions = data.map(lambda line: line.strip().split(' '))
print transactions.take(5)
model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()
for fi in result:
    print(fi)

for i in result:
    print '(', ', '.join(i.items), ')', 'freq=', str(i.freq)

#############################################

data = [["a", "b", "c"], ["a", "b", "d", "e"], ["a", "c", "e"], ["a", "c", "f"]]
rdd = sc.parallelize(data, 2)
model = FPGrowth.train(rdd, 0.6, 2)
sorted(model.freqItemsets().collect())

####################################################

# Model fitted by PrefixSpan

data = [
[["a", "b"], ["c"]],
[["a"], ["c", "b"], ["a", "b"]],
[["a", "b"], ["e"]],
[["f"]]]

rdd = sc.parallelize(data)
model = PrefixSpan.train(rdd)
sorted(model.freqSequences().collect())

######################################

rdd = sc.parallelize(data)
print rdd.collect()
model = PrefixSpan.train(rdd)
sorted(model.freqSequences().collect())

from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.types import *
from datetime import datetime
import time
from pyspark.sql.functions import *

# sc is an existing SparkContext.
sqlContext = HiveContext(sc)

data = sc.parallelize([[1, "v1", "d1"], [1, "v2", "d2"], [2, "v21", "d21"], [2, "v22", "d22"]])

data = sc.parallelize([[1, "v1"], [1, "v2"], [2, "v21"], [2, "v22"]])

model = PrefixSpan.train(data)
sorted(model.freqSequences().collect())

#################################

from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.types import *
from datetime import datetime
import time
from pyspark.sql.functions import *
from pyspark.mllib.fpm import FPGrowth, PrefixSpan

# sc is an existing SparkContext.
sqlContext = HiveContext(sc)
#lines = sc.textFile("hdfs:///tmp/fact_icd9_encounter_08242016.txt", 20)

lines = sc.textFile("/Users/jayurbain/Dropbox/MCW/fact_icd9_encounter_08242016.txt")

parts = lines.map(lambda l: l.split("\t"))
encounters = parts.map(lambda p: (p[0], datetime.strptime(p[1], "%Y-%m-%d %H:%M:%S"), p[2], p[3].strip().replace(',','_')))

fields = [StructField("PATIENT_NUM", StringType(), True),
	StructField("START_DATE", DateType(), True),
	StructField("ENCOUNTER_NUM", StringType(), True),
	StructField("ICD9S", StringType(), True)]
schema_encounters = StructType(fields)

# fields = [StructField("PATIENT_NUM", StringType(), True),
# 	StructField("ENCOUNTER_NUM", StringType(), True),
# 	StructField("START_DATE", StringType(), True),
# 	Seq(StructField("ICD9S", ArrayType(StringType(), True), True) ]
# schema_encounters = StructType(fields)

# Apply the schema to the RDD.
schemaEncounters = sqlContext.createDataFrame(encounters, schema_encounters)
schemaEncounters.printSchema()
schemaEncounters.registerTempTable("encounters")

# order data by patient, start date, _then_ encounter

encounteres_ordered = sqlContext.sql("select PATIENT_NUM, START_DATE, ENCOUNTER_NUM, ICD9S from encounters order by PATIENT_NUM, START_DATE, ENCOUNTER_NUM")
encounteres_ordered.registerTempTable("encounteres_ordered")

#sqlContext.sql("select collect_list(ICD9S) as icd9s from encounteres_ordered group by PATIENT_NUM").show(5)

#sqlContext.sql("select PATIENT_NUM, collect_list(ICD9S) as icd9s from encounteres_ordered group by PATIENT_NUM").show(20, truncate=False)

rdd = sqlContext.sql("select collect_list(ICD9S) as icd9s from encounteres_ordered group by PATIENT_NUM").rdd

def splitter(p):
	li = list()
	for i in p:
		l = i.split('_')
		li.append(l)
	return li

seqrdd = rdd.map(lambda x: splitter(x[0]) )
seqrdd.getNumPartitions()
#seqrdd.cache()

# minSupport=0.1, maxPatternLength=10, maxLocalProjDBSize=32000000)
model = PrefixSpan.train(seqrdd, minSupport=0.05, maxPatternLength=10)
#sorted(model.freqSequences().collect(), reverse=True)

# sorted(model.freqSequences(), key=lambda x: x[1])

model.freqSequences().takeOrdered(1000, key=lambda x: -x[1])

model_df = model.freqSequences().toDF

result_ps = model.freqSequences().collect()
result_ps = model.freqSequences().take(100)
for i in sorted(result_ps, key=operator.itemgetter(1), reverse=True):
    print '(', i.sequence, ')', 'freq=', str(i.freq)

freqSeqDf = model.freqSequences().collect().toDF(["seq","freq"])

#model.freqSequences().saveAsTextFile("hdfs:///tmp/fact_icd9_encounter_08242016_supP2_rdd.txt")

#lines = sc.textFile("/Users/jayurbain/Dropbox/MCW/fact_icd9_encounter_08242016.txt")

model.freqSequences().saveAsTextFile("/Users/jayurbain/Dropbox/MCW/fact_icd9_encounter_08242016_supP2_rdd.txt")
###################################

# DateType TimestampType
fields = [StructField("PATIENT_NUM", StringType(), True),
	StructField("VITAL_STATUS_CD", StringType(), True),
	StructField("BIRTH_DATE", DateType(), True),
	StructField("DEATH_DATE", DateType(), True),
	StructField("SEX_CD", StringType(), True),
	StructField("AGE_IN_YEARS_NUM", IntegerType(), True),
	StructField("LANGUAGE_CD", StringType(), True),
	StructField("RACE_CD", StringType(), True),
	StructField("MARITAL_STATUS_CD", StringType(), True),
	StructField("RELIGION_CD", StringType(), True),
	StructField("ZIP_CD", StringType(), True)]
schema_patients = StructType(fields)

# Apply the schema to the RDD.
schemaPatients = sqlContext.createDataFrame(patients, schema_patients)

schemaPatients.columns

schemaPatients.printSchema()


###################################

def toCSVLine(data):
  return ','.join(str(d) for d in data)

linesOut = model.freqSequences().map(toCSVLine)

#linesOut.saveAsTextFile("hdfs:///tmp/fact_icd9_encounter_08242016_supP2_rdd_.csv")

# linesOut.saveAsTextFile('hdfs:///tmp/fact_icd9_encounter_08242016_supP2_rdd.csv')

linesOut.saveAsTextFile("/Users/jayurbain/Dropbox/MCW/fact_icd9_encounter_08242016_supP2_rdd.txt")

linesOut.reduce( lambda k,v: (k))

################

from pyspark.sql import Row
from pyspark.sql.functions import explode

df = sqlContext.createDataFrame([Row(a=1, b=[1,2,3],c=[7,8,9]), Row(a=2, b=[4,5,6],c=[10,11,12])])
df1 = df.select(df.a,explode(df.b).alias("b"),df.c)
df2 = df1.select(df1.a,df1.b,explode(df1.c).alias("c"))



###################################

# fpgrowth example
itemsets = parts.map(lambda p: ( p[3].strip().split(',') ) )
itemsets.getNumPartitions()
model_fp = FPGrowth.train(itemsets, minSupport=0.005, numPartitions=10)
result = model_fp.freqItemsets().collect()
for i in sorted(result, key=operator.itemgetter(1), reverse=True):
    print '(', ', '.join(i.items), ')', 'freq=', str(i.freq)


# ICD-9-CM V58.69 converts approximately to: 2016 ICD-10-CM Z79.891 Long term (current) use of opiate analgesic.

