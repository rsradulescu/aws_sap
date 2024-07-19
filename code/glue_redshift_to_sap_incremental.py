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

# Retrieve the credentials for sap bw & redshift user from the secret
def get_secret(secret_name, region_name):
    """Retrieve the secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Read data from redshift as Dataframe
def read_from_redshift(host, port, db, schema_name, table_name, secret_name, region_name):
    """Read data from Redshift."""
    try:
        # Retrieve the db credentials
        secrets = get_secret(secret_name, region_name)
        uid = secrets.get("redshift_uid")
        pwd = secrets.get("redshift_pwd")

        url = f"jdbc:redshift://{host}:{port}/{db}"
        df = glueContext.read \
                        .format("jdbc") \
                        .option("url", url) \
                        .option("currentschema", schema_name) \
                        .option("dbtable", table_name) \
                        .option("user", uid) \
                        .option("password", pwd) \
                        .load()

        return df
    except Exception as e:
        print(f"Error reading from Redshift: {str(e)}")

# Read data from sap bw system as dataframe
def read_from_sap_hana_sampledb(db_url, schema, view_name, uid, pwd):
    """Read data from SAP HANA sampledb source."""
    df = glueContext.read \
                  .format("jdbc") \
                  .option("driver", "com.sap.db.jdbc.Driver") \
                  .option("url", db_url) \
                  .option("currentschema", schema) \
                  .option("dbtable", view_name) \
                  .option("user", uid) \
                  .option("password", pwd) \
                  .load()
    return df
    
    
def transform_data(df):
    """Transform the data."""
    # Get rid of pre-aggregated columns
    df = df.drop("exchange_rate_diff_from_avg") \
                   .drop("insert_timestamp")
                   
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
            .mode("append") \
            .save()
        print("Data loaded to SAP HANA sampledb successfully!")
    except Exception as e:
        print(f"Error loading data to SAP HANA sampledb: {str(e)}")

# Apply incremental load logic        
def incremental_load_to_sap(schema_name,table_name,uid,pwd,redshift_df,sap_url):
    """Write the most recent records in DataFrame to SAP HANA sampledb system using incremental strategy."""
    try:
        # Read the sap hana target table where we will apply incremental load
        sap_df = read_from_sap_hana_sampledb(sap_url, schema_name, table_name, uid, pwd)
        
        # Find the maximum exchange_date in the sap table
        max_sap_date = sap_df.agg(F.max(F.col("exchange_date"))).collect()[0][0]

        # Use Spark SQL expressions to filter SAP data
        filtered_redshift_df = redshift_df.filter(
            F.col("exchange_date").cast("date") > F.lit(max_sap_date)
        )

        if filtered_redshift_df.count() == 0:
            print("No new or updated records were found for loading into the sap hana table")
        else:
            print("Data that will be loaded to sap hana table:")
            filtered_redshift_df.show(5)
            
            # Perform the incremental loading
            load_to_sap_hana_sampledb(filtered_redshift_df, sap_url, schema_name, table_name, uid, pwd)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
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
    
    # Define the connection string of redshift source
    host = "redshift_host"
    port = "redshift_port"
    db = "redshift_db"
    schema_name = "redshift_schema"
    # Define the name of the new redshift table
    redshift_table_name = "target_table"

    # Read data from redshift (source)
    redshift_df = read_from_redshift(host, port, db, schema_name, redshift_table_name, secret_name, region_name)
 
    # Transform the data
    final_df = transform_data(redshift_df)
    
    # Define the connection string of SAP HANA target
    db_url = 'jdbc:sap://10.00.00.00:9000/?databaseName=sampledb'
    # Make sure to create the target table on sap system first, otherwise write operation will fail
    sap_table_name = "sap_table_name" 
    sap_schema = "schema_name" 
    
    # Load most recent data from redshift to sap hana sampledb using incremental load
    incremental_load_to_sap(sap_schema,sap_table_name,uid,pwd,final_df,db_url)
    
 
    print("Job has finished!")
    job.commit()

if __name__ == "__main__":
    main()
