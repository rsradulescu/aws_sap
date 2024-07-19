import sys
import re
from awsglue.transforms import *
from datetime import datetime
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import current_timestamp
import boto3
import json
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Get AWS Glue job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Retrieve the credentials for sap bw  from the secret
def get_secret(secret_name, region_name):
    """Retrieve the secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Read data from s3 (source)
def read_from_s3(s3_base_path):
    """Read data from Redshift."""
    try:
        # Read data from s3's partitioned folders with the syntax of "year=2024/month=1/day=1/file_name.snappy.parquet"
        df = spark.read.parquet(s3_base_path+"/*/*/*/*.parquet")
        # Convert the dynamic frame to a Spark DataFrame
        df.printSchema()
        df.show(5)
        return df
    except Exception as e:
        print(f"Error reading from S3: {str(e)}")
    
def transform_data(df):
    """Transform the data."""
    # Get rid of pre-aggregated columns
    df = df.drop("exchange_rate_diff_from_avg") \
                   .drop("lag_exchange_rate") \
                   
     # Drop null values
    df = df.dropna()

    # Remove duplicates
    df = df.dropDuplicates()

    # Define a window specification to calculate aggregate values
    windowSpec = Window.partitionBy("exchange_rate_type", "curr_code_from_cntry", "curr_code_target_cntry")

    # Calculate aggregate values using window functions
    final_df = df.withColumn("avg_exchange_rate", F.avg("exchange_rate").over(windowSpec)) \
                 .withColumn("max_exchange_rate", F.max("exchange_rate").over(windowSpec)) \
                 .withColumn("min_exchange_rate", F.min("exchange_rate").over(windowSpec))
     
    return final_df

# Write data to sap bw system 
def load_to_sap_hana_sampledb(df, db_url, schema, table_name, uid, pwd):
    """Write DataFrame to SAP HANA sampledb."""
    try:
        # Add insert timestamp column
        df= df.withColumn("insert_timestamp", current_timestamp())
        
        # Write data
        df.write \
            .format("jdbc") \
            .option("driver", "com.sap.db.jdbc.Driver") \
            .option("url", db_url) \
            .option("currentschema", schema) \
            .option("dbtable", table_name) \
            .option("user", uid) \
            .option("password", pwd) \
            .mode("overwrite") \
            .save()
        print("Data loaded to SAP HANA sampledb successfully!")
    except Exception as e:
        print(f"Error loading data to SAP HANA sampledb: {str(e)}")
        

def main():
    """Main function to orchestrate the ETL process."""
    job.init(args['JOB_NAME'], args)
    
    # Define the secret variables
    secret_name = "secret_name"
    region_name = "secret_region"

    # Retrieve the db credentials
    secrets = get_secret(secret_name, region_name)
    uid = secrets.get("uid")
    pwd = secrets.get("pwd")

    # Define the connection string of SAP HANA target
    db_url = 'jdbc:sap://10.00.00.00:9000/?databaseName=sampledb'
    # Make sure to create the target table on sap system first, otherwise write operation will fail
    sap_table_name = "target_sap_table_name" 
    sap_schema = "schema_name" 
    
    # Define s3 source path
    s3_base_path = "s3://bucket_name/sap_product/sub_folder"
    
    # Read data from s3 
    df = read_from_s3(s3_base_path)
    
    # Transform the data
    final_df = transform_data(df)
    
    # Load all the data from s3 to sap hana sampledb 
    load_to_sap_hana_sampledb(final_df, db_url, sap_schema, sap_table_name, uid, pwd)
    
 
    print("Job has finished!")
    job.commit()

if __name__ == "__main__":
    main()
