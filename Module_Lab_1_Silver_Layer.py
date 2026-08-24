# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup catalog and schema variables
# Databricks widgets for portable configuration
dbutils.widgets.text("catalog", "workspace", "Catalog Name")
dbutils.widgets.text("schema_bronze", "cbl_bronze", "Bronze Schema")
dbutils.widgets.text("schema_silver", "cbl_silver", "Silver Schema")
dbutils.widgets.text("schema_gold", "cbl_gold", "Gold Schema")

# Read widget values into variables
catalog = dbutils.widgets.get("catalog")
schema_bronze = dbutils.widgets.get("schema_bronze")
schema_silver = dbutils.widgets.get("schema_silver")
schema_gold = dbutils.widgets.get("schema_gold")

# Print configuration
print("Data Layer Configuration:")
print(f"  Catalog: {catalog}")
print(f"  Bronze Schema: {schema_bronze}")
print(f"  Silver Schema: {schema_silver}")
print(f"  Gold Schema: {schema_gold}")
print()

# Construct bronze table names
table_sap_billing = f"{catalog}.{schema_bronze}.sap_billing"
table_kna1_customer = f"{catalog}.{schema_bronze}.kna1_customer"
table_mara_material = f"{catalog}.{schema_bronze}.mara_material"

print("Bronze Tables:")
print(f"  SAP Billing: {table_sap_billing}")
print(f"  Customer: {table_kna1_customer}")
print(f"  Material: {table_mara_material}")

# COMMAND ----------

# DBTITLE 1,Read and clean SAP billing data
from pyspark.sql.functions import col, to_date, trim, when, upper, coalesce, regexp_replace, lpad, length, expr, lit

# Read bronze SAP billing table
print(f"Reading from: {table_sap_billing}")
df_billing_bronze = spark.table(table_sap_billing)

# Silver layer transformations:
# 1. Convert date columns to proper date type (handle 3 formats)
# 2. Clean European decimal format (dot=thousands, comma=decimal)
# 3. Handle placeholder values before casting
# 4. Normalize currency codes
# 5. Trim and clean string columns

df_billing_silver = (df_billing_bronze
    # Convert date fields - handle 3 formats: yyyyMMdd, dd.MM.yyyy, yyyy-MM-dd
    # Use try_to_date to return NULL on parse failure instead of error
    .withColumn("billing_date", 
                coalesce(
                    expr("try_to_date(FKDAT, 'yyyyMMdd')"),
                    expr("try_to_date(FKDAT, 'dd.MM.yyyy')"),
                    expr("try_to_date(FKDAT, 'yyyy-MM-dd')")
                ))
    .withColumn("created_date", 
                coalesce(
                    expr("try_to_date(ERDAT, 'yyyyMMdd')"),
                    expr("try_to_date(ERDAT, 'dd.MM.yyyy')"),
                    expr("try_to_date(ERDAT, 'yyyy-MM-dd')")
                ))
    .withColumn("changed_date", 
                coalesce(
                    expr("try_to_date(AEDAT, 'yyyyMMdd')"),
                    expr("try_to_date(AEDAT, 'dd.MM.yyyy')"),
                    expr("try_to_date(AEDAT, 'yyyy-MM-dd')")
                ))
    
    # Cast numeric columns with European decimal format handling
    # FKIMG: European format (dot=thousands, comma=decimal) -> "5.602,68" becomes 5602.68
    .withColumn("quantity", 
                regexp_replace(regexp_replace(col("FKIMG"), "\\.", ""), ",", ".").cast("decimal(15,3)"))
    
    # NETWR: European format (dot=thousands, comma=decimal)
    .withColumn("net_value", 
                regexp_replace(regexp_replace(col("NETWR"), "\\.", ""), ",", ".").cast("decimal(15,2)"))
    
    # MWSBP: Map placeholder strings to NULL before casting
    .withColumn("tax_amount", 
                when(col("MWSBP").isin("#N/A", "NULL", "-", "n/a", ""), None)
                .otherwise(regexp_replace(regexp_replace(col("MWSBP"), "\\.", ""), ",", "."))
                .cast("decimal(15,2)"))
    
    # ZZUNITPRICE: European decimal format
    .withColumn("unit_price", 
                regexp_replace(regexp_replace(col("ZZUNITPRICE"), "\\.", ""), ",", ".").cast("decimal(15,2)"))
    
    # ZZDISCPCT: European decimal format
    .withColumn("discount_pct", 
                regexp_replace(regexp_replace(col("ZZDISCPCT"), "\\.", ""), ",", ".").cast("decimal(5,2)"))
    
    # Trim and clean string columns
    .withColumn("client", trim(col("MANDT")))
    .withColumn("billing_doc", trim(col("VBELN")))
    .withColumn("line_item", trim(col("POSNR")))
    .withColumn("billing_type", trim(col("FKART")))
    .withColumn("customer_id", trim(col("KUNNR")))
    
    # MATNR: Left-pad to 18 characters with zeros
    .withColumn("material_id", lpad(trim(col("MATNR")), 18, "0"))
    
    .withColumn("plant", trim(col("WERKS")))
    .withColumn("distribution_channel", trim(col("VTWEG")))
    .withColumn("division", trim(col("SPART")))
    .withColumn("vendor_id", trim(col("LIFNR")))
    
    # VRKME: Uppercase and treat blank as NULL
    .withColumn("unit_of_measure", 
                when(trim(col("VRKME")) == "", None)
                .otherwise(upper(trim(col("VRKME")))))
    
    # WAERK: Normalize currency codes - lkr, "LKR ", Rs. all become LKR
    .withColumn("currency", 
                when(trim(upper(col("WAERK"))).isin("LKR", "RS."), "LKR")
                .otherwise(trim(upper(col("WAERK")))))
    
    .withColumn("payment_terms", trim(col("ZTERM")))
    .withColumn("order_channel", upper(trim(col("ZZORDCHAN"))))
    .withColumn("loyalty_id", trim(col("ZZLOYALTY_ID")))
    .withColumn("promo_code", trim(col("ZZPROMO_CODE")))
    
    # Add data quality flag (set to False since _rescued_data column doesn't exist)
    .withColumn("has_data_quality_issues", lit(False))
    
    # Select cleaned columns
    .select(
        "client", "billing_doc", "line_item", "billing_type",
        "billing_date", "created_date", "changed_date",
        "customer_id", "material_id", "plant",
        "distribution_channel", "division", "vendor_id",
        "quantity", "unit_of_measure", 
        "net_value", "tax_amount", "currency",
        "unit_price", "discount_pct",
        "payment_terms", "order_channel",
        "loyalty_id", "promo_code",
        "has_data_quality_issues",
        "_ingest_timestamp", "_batch_id"
    )
)

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

# Count rows before deduplication
rows_before_dedup = df_billing_silver.count()

# Step 1: Remove exact duplicates across all columns
df_billing_silver_dedup = df_billing_silver.distinct()
rows_after_exact_dedup = df_billing_silver_dedup.count()
exact_duplicates_removed = rows_before_dedup - rows_after_exact_dedup

# Step 2: Remove business key duplicates - keep only latest AEDAT per (VBELN, POSNR)
# Define window: partition by business key, order by changed_date descending (NULL last)
window_spec = Window.partitionBy("billing_doc", "line_item") \
                    .orderBy(desc("changed_date"))

# Add row number within each business key group
df_with_row_num = df_billing_silver_dedup.withColumn("row_num", row_number().over(window_spec))

# Keep only the latest record (row_num = 1) per business key
df_billing_silver_final = df_with_row_num.filter(col("row_num") == 1).drop("row_num")

rows_after_business_key_dedup = df_billing_silver_final.count()
business_key_duplicates_removed = rows_after_exact_dedup - rows_after_business_key_dedup

# Step 3: Exclude test data (customer_id = 'ZZTEST0001' with quantity 9999)
df_no_test_data = df_billing_silver_final.filter(
    ~((col("customer_id") == "ZZTEST0001") & (col("quantity") == 9999))
)
rows_after_test_exclusion = df_no_test_data.count()
test_rows_removed = rows_after_business_key_dedup - rows_after_test_exclusion

# Step 4: Split into invoices (F2) and credit memos (G2)
df_invoices = df_no_test_data.filter(col("billing_type") == "F2")
df_credit_memos = df_no_test_data.filter(col("billing_type") == "G2")

invoice_count = df_invoices.count()
credit_memo_count = df_credit_memos.count()
other_types_count = rows_after_test_exclusion - invoice_count - credit_memo_count

print(f"Bronze rows: {df_billing_bronze.count():,}")
print(f"\nData Cleaning Summary:")
print(f"  Starting rows: {rows_before_dedup:,}")
print(f"  After exact duplicate removal: {rows_after_exact_dedup:,} (removed {exact_duplicates_removed:,})")
print(f"  After business key deduplication: {rows_after_business_key_dedup:,} (removed {business_key_duplicates_removed:,})")
print(f"  After test data exclusion: {rows_after_test_exclusion:,} (removed {test_rows_removed:,})")
print(f"\n📊 Document Type Split:")
print(f"  F2 Invoices: {invoice_count:,} rows → fact_billing")
print(f"  G2 Credit Memos: {credit_memo_count:,} rows → fact_credit_memo")
if other_types_count > 0:
    print(f"  Other types: {other_types_count:,} rows (excluded)")
print(f"\n🗑️  Total rows removed: {rows_before_dedup - rows_after_test_exclusion:,} ({(rows_before_dedup - rows_after_test_exclusion)/rows_before_dedup*100:.2f}%)")
print(f"     - Exact duplicates: {exact_duplicates_removed:,}")
print(f"     - Business key duplicates (kept latest AEDAT): {business_key_duplicates_removed:,}")
print(f"     - Test data (ZZTEST0001 with qty 9999): {test_rows_removed:,}")
print(f"\nSample invoice data:")
display(df_invoices.limit(5))

# Update the variable names for downstream use
df_billing_silver = df_invoices
df_credit_memo_silver = df_credit_memos

# COMMAND ----------

# DBTITLE 1,Check for NULL billing dates and key fields
from pyspark.sql.functions import col, count, when

print("NULL Value Analysis\n")
print("=" * 60)

total_rows = df_billing_silver.count()
print(f"Total rows in silver table: {total_rows:,}\n")

# Count NULL values in key columns
null_counts = df_billing_silver.select(
    count(when(col("billing_date").isNull(), 1)).alias("null_billing_dates"),
    count(when(col("created_date").isNull(), 1)).alias("null_created_dates"),
    count(when(col("changed_date").isNull(), 1)).alias("null_changed_dates"),
    count(when(col("quantity").isNull(), 1)).alias("null_quantities"),
    count(when(col("net_value").isNull(), 1)).alias("null_net_values"),
    count(when(col("tax_amount").isNull(), 1)).alias("null_tax_amounts"),
    count(when(col("unit_of_measure").isNull(), 1)).alias("null_uom"),
    count(when(col("currency").isNull(), 1)).alias("null_currency")
).collect()[0]

print("NULL Value Summary:")
print(f"  Billing Date:       {null_counts['null_billing_dates']:,} ({null_counts['null_billing_dates']/total_rows*100:.2f}%)")
print(f"  Created Date:       {null_counts['null_created_dates']:,} ({null_counts['null_created_dates']/total_rows*100:.2f}%)")
print(f"  Changed Date:       {null_counts['null_changed_dates']:,} ({null_counts['null_changed_dates']/total_rows*100:.2f}%)")
print(f"  Quantity:           {null_counts['null_quantities']:,} ({null_counts['null_quantities']/total_rows*100:.2f}%)")
print(f"  Net Value:          {null_counts['null_net_values']:,} ({null_counts['null_net_values']/total_rows*100:.2f}%)")
print(f"  Tax Amount:         {null_counts['null_tax_amounts']:,} ({null_counts['null_tax_amounts']/total_rows*100:.2f}%)")
print(f"  Unit of Measure:    {null_counts['null_uom']:,} ({null_counts['null_uom']/total_rows*100:.2f}%)")
print(f"  Currency:           {null_counts['null_currency']:,} ({null_counts['null_currency']/total_rows*100:.2f}%)")

# Show sample records with null billing dates if any exist
if null_counts['null_billing_dates'] > 0:
    print(f"\nSample records with NULL billing_date:")
    null_billing_samples = df_billing_silver.filter(col("billing_date").isNull()).limit(5)
    display(null_billing_samples.select("billing_doc", "line_item", "billing_date", "created_date", "customer_id", "material_id", "net_value"))

# COMMAND ----------

# DBTITLE 1,Verify data quality transformations
from pyspark.sql import Window
from pyspark.sql.functions import count, col, row_number

print("Duplicate Data Analysis\n")
print("=" * 60)

# 1. Check for exact duplicate rows
print("\n1. Exact Duplicate Rows:")
total_rows = df_billing_silver.count()
unique_rows = df_billing_silver.distinct().count()
exact_duplicates = total_rows - unique_rows
print(f"   Total rows: {total_rows:,}")
print(f"   Unique rows: {unique_rows:,}")
print(f"   Exact duplicates: {exact_duplicates:,}")

# 2. Check for duplicate business keys (billing_doc + line_item)
print("\n2. Duplicate Business Keys (billing_doc + line_item):")
business_key_dups = df_billing_silver.groupBy("billing_doc", "line_item") \
    .count() \
    .filter(col("count") > 1) \
    .orderBy(col("count").desc())

dup_count = business_key_dups.count()
print(f"   Duplicate business key combinations: {dup_count:,}")

if dup_count > 0:
    print(f"\n   Top 10 most duplicated business keys:")
    display(business_key_dups.limit(10))
    
    # Show sample duplicates
    print("\n   Sample duplicate records:")
    sample_dup = business_key_dups.limit(1).collect()[0]
    sample_doc = sample_dup['billing_doc']
    sample_item = sample_dup['line_item']
    
    duplicate_records = df_billing_silver \
        .filter((col("billing_doc") == sample_doc) & (col("line_item") == sample_item)) \
        .orderBy("billing_date")
    
    print(f"\n   Example: billing_doc={sample_doc}, line_item={sample_item}")
    display(duplicate_records)
else:
    print("   ✓ No duplicate business keys found!")

# 3. Check for duplicate billing documents (multiple line items per doc)
print("\n3. Billing Documents Statistics:")
lines_per_doc = df_billing_silver.groupBy("billing_doc") \
    .agg(count("*").alias("line_count")) \
    .groupBy("line_count") \
    .count() \
    .orderBy("line_count")

print("   Distribution of line items per billing document:")
display(lines_per_doc)

# COMMAND ----------

# DBTITLE 1,Read and clean customer master data
from pyspark.sql.functions import col, to_date, trim, when, coalesce, expr, regexp_replace

# Read bronze customer table
print(f"Reading from: {table_kna1_customer}")
df_customer_bronze = spark.table(table_kna1_customer)

# Silver layer transformations for customer data
df_customer_silver = (df_customer_bronze
    # Convert date field - handle 3 formats: yyyyMMdd, dd.MM.yyyy, yyyy-MM-dd
    .withColumn("created_date", 
                coalesce(
                    expr("try_to_date(ERDAT, 'yyyyMMdd')"),
                    expr("try_to_date(ERDAT, 'dd.MM.yyyy')"),
                    expr("try_to_date(ERDAT, 'yyyy-MM-dd')")
                ))
    
    # Cast numeric/decimal columns - handle European decimal format for credit_limit
    .withColumn("credit_limit", 
                regexp_replace(regexp_replace(col("KLIMK"), "\\.", ""), ",", ".").cast("decimal(15,2)"))
    .withColumn("cooler_count", col("ZZCOOLERS").cast("int"))
    .withColumn("latitude", col("ZZLAT").cast("decimal(10,7)"))
    .withColumn("longitude", col("ZZLON").cast("decimal(10,7)"))
    
    # Trim and clean string columns
    .withColumn("client", trim(col("MANDT")))
    .withColumn("customer_id", trim(col("KUNNR")))
    .withColumn("customer_name", trim(col("NAME1")))
    .withColumn("street_address", trim(col("STRAS")))
    .withColumn("city", trim(col("ORT01")))
    .withColumn("region", trim(col("REGIO")))
    .withColumn("account_group", trim(col("KTOKD")))
    .withColumn("customer_tier", trim(col("ZZTIER")))
    .withColumn("area_type", trim(col("ZZAREATYPE")))
    .withColumn("is_exclusive", trim(col("ZZEXCLUSIVE")))
    .withColumn("deletion_flag", trim(col("LOEVM")))
    
    # Add active status flag
    .withColumn("is_active", 
                when(col("deletion_flag").isNull() | (col("deletion_flag") == ""), True).otherwise(False))
    
    # Select cleaned columns
    .select(
        "client", "customer_id", "customer_name",
        "street_address", "city", "region",
        "account_group", "customer_tier", "area_type",
        "created_date", "credit_limit",
        "cooler_count", "is_exclusive",
        "latitude", "longitude",
        "is_active", "deletion_flag",
        "_ingest_timestamp", "_batch_id"
    )
)

print(f"Bronze rows: {df_customer_bronze.count():,}")
print(f"Silver rows: {df_customer_silver.count():,}")
print(f"\nSample data:")
display(df_customer_silver.limit(5))

# COMMAND ----------

# DBTITLE 1,Read and clean material master data
from pyspark.sql.functions import col, to_date, trim, upper, coalesce, expr, regexp_replace

# Read bronze material table
print(f"Reading from: {table_mara_material}")
df_material_bronze = spark.table(table_mara_material)

# Silver layer transformations for material data
df_material_silver = (df_material_bronze
    # Convert date field - handle 3 formats: yyyyMMdd, dd.MM.yyyy, yyyy-MM-dd
    .withColumn("launch_date", 
                coalesce(
                    expr("try_to_date(ZZLAUNCHDATE, 'yyyyMMdd')"),
                    expr("try_to_date(ZZLAUNCHDATE, 'dd.MM.yyyy')"),
                    expr("try_to_date(ZZLAUNCHDATE, 'yyyy-MM-dd')")
                ))
    
    # Cast numeric columns - handle European decimal format where applicable
    .withColumn("abv_percentage", 
                regexp_replace(regexp_replace(col("ZZABV"), "\\.", ""), ",", ".").cast("decimal(5,2)"))
    .withColumn("pack_ml", col("ZZPACKML").cast("int"))
    .withColumn("units_per_case", col("ZZUNITSCASE").cast("int"))
    .withColumn("list_price", 
                regexp_replace(regexp_replace(col("ZZLISTPRICE"), "\\.", ""), ",", ".").cast("decimal(15,2)"))
    
    # Trim and clean string columns
    .withColumn("material_id", trim(col("MATNR")))
    .withColumn("material_description", trim(col("MAKTX")))
    .withColumn("material_group", trim(col("MATKL")))
    .withColumn("base_unit_of_measure", trim(col("MEINS")))
    .withColumn("brand", trim(col("ZZBRAND")))
    .withColumn("category", trim(col("ZZCATEGORY")))
    .withColumn("pack_type", upper(trim(col("ZZPACKTYPE"))))
    
    # Select cleaned columns
    .select(
        "material_id", "material_description",
        "material_group", "base_unit_of_measure",
        "brand", "category",
        "abv_percentage", "pack_ml", "pack_type",
        "units_per_case", "list_price",
        "launch_date",
        "_ingest_timestamp", "_batch_id"
    )
)

print(f"Bronze rows: {df_material_bronze.count():,}")
print(f"Silver rows: {df_material_silver.count():,}")
print(f"\nSample data:")
display(df_material_silver.limit(5))

# COMMAND ----------

# DBTITLE 1,Write to silver layer tables
# Define silver table names using variables
silver_billing = f"{catalog}.{schema_silver}.fact_billing"
silver_credit_memo = f"{catalog}.{schema_silver}.fact_credit_memo"
silver_customer = f"{catalog}.{schema_silver}.kna1_customer"
silver_material = f"{catalog}.{schema_silver}.mara_material"

print("Writing to Silver Layer:")
print()

# Write SAP Billing Invoices (F2)
print(f"1. Writing {df_billing_silver.count():,} F2 invoice rows to {silver_billing}...")
df_billing_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(silver_billing)
print("   ✓ Complete")

# Write Credit Memos (G2)
print(f"2. Writing {df_credit_memo_silver.count():,} G2 credit memo rows to {silver_credit_memo}...")
df_credit_memo_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(silver_credit_memo)
print("   ✓ Complete")

# Write Customer
print(f"3. Writing {df_customer_silver.count():,} rows to {silver_customer}...")
df_customer_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(silver_customer)
print("   ✓ Complete")

# Write Material
print(f"4. Writing {df_material_silver.count():,} rows to {silver_material}...")
df_material_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(silver_material)
print("   ✓ Complete")

print()
print("Silver layer transformation completed successfully!")

# COMMAND ----------

# DBTITLE 1,Verify silver layer tables
# Summary of all silver tables
query = f"""
SELECT 
  'fact_billing' as table_name,
  COUNT(*) as row_count,
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(DISTINCT material_id) as unique_materials,
  ROUND(SUM(net_value), 2) as total_net_value
FROM {catalog}.{schema_silver}.fact_billing

UNION ALL

SELECT 
  'fact_credit_memo' as table_name,
  COUNT(*) as row_count,
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(DISTINCT material_id) as unique_materials,
  ROUND(SUM(net_value), 2) as total_net_value
FROM {catalog}.{schema_silver}.fact_credit_memo

UNION ALL

SELECT 
  'kna1_customer' as table_name,
  COUNT(*) as row_count,
  SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_customers,
  NULL as unique_materials,
  NULL as total_net_value
FROM {catalog}.{schema_silver}.kna1_customer

UNION ALL

SELECT 
  'mara_material' as table_name,
  COUNT(*) as row_count,
  COUNT(DISTINCT brand) as unique_brands,
  COUNT(DISTINCT category) as unique_categories,
  NULL as total_net_value
FROM {catalog}.{schema_silver}.mara_material

ORDER BY table_name
"""

result = spark.sql(query)
display(result)

# COMMAND ----------

# DBTITLE 1,Analyze FKDAT date format patterns
# MAGIC %sql
# MAGIC -- Show 5 distinct raw string values from FKIMG (quantity)
# MAGIC SELECT DISTINCT FKIMG
# MAGIC FROM workspace.cbl_bronze.sap_billing
# MAGIC WHERE FKIMG IS NOT NULL
# MAGIC LIMIT 5;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Build dim_product from mara_material
from pyspark.sql.functions import lpad, current_timestamp, lit
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_bronze}.mara_material"
target_table = f"{catalog}.{schema_silver}.dim_product"

print(f"Building dimension table: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read bronze material master
df_material = spark.table(source_table)

print(f"Bronze materials: {df_material.count():,} rows\n")

# Transform to dimension with left-padded product_id
df_dim_product = (df_material
    .withColumn("product_id", lpad(df_material.MATNR, 18, "0"))  # Left-pad to 18 chars
    .withColumnRenamed("MATNR", "material_code_raw")
    .withColumnRenamed("MAKTX", "product_description")
    .withColumnRenamed("MATKL", "material_group")
    .withColumnRenamed("MEINS", "base_unit_of_measure")
    .withColumnRenamed("ZZBRAND", "brand")
    .withColumnRenamed("ZZCATEGORY", "category")
    .withColumnRenamed("ZZABV", "alcohol_by_volume")
    .withColumnRenamed("ZZPACKML", "package_size_ml")
    .withColumnRenamed("ZZPACKTYPE", "package_type")
    .withColumnRenamed("ZZUNITSCASE", "units_per_case")
    .withColumnRenamed("ZZLISTPRICE", "list_price")
    .withColumnRenamed("ZZLAUNCHDATE", "launch_date")
    .withColumn("_ingest_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
    .select(
        "product_id",
        "material_code_raw",
        "product_description",
        "brand",
        "category",
        "material_group",
        "base_unit_of_measure",
        "alcohol_by_volume",
        "package_size_ml",
        "package_type",
        "units_per_case",
        "list_price",
        "launch_date",
        "_ingest_timestamp",
        "_batch_id"
    )
)

print("Sample transformed data:")
display(df_dim_product.select("product_id", "material_code_raw", "product_description", "brand", "category").limit(5))

# Write to silver with overwrite
df_dim_product.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

row_count = df_dim_product.count()
print(f"\n✓ Dimension table created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {row_count:,}")

# COMMAND ----------

# DBTITLE 1,Verify all products have sales
# MAGIC %sql
# MAGIC -- Verify sales coverage: Check which products in dim_product have sales in fact_billing
# MAGIC SELECT 
# MAGIC   'Total products in dimension' as metric,
# MAGIC   COUNT(DISTINCT d.product_id) as count
# MAGIC FROM workspace.cbl_silver.dim_product d
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'Products with sales' as metric,
# MAGIC   COUNT(DISTINCT d.product_id) as count
# MAGIC FROM workspace.cbl_silver.dim_product d
# MAGIC INNER JOIN workspace.cbl_silver.fact_billing b
# MAGIC   ON d.product_id = b.material_id
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'Products WITHOUT sales' as metric,
# MAGIC   COUNT(DISTINCT d.product_id) as count
# MAGIC FROM workspace.cbl_silver.dim_product d
# MAGIC LEFT JOIN workspace.cbl_silver.fact_billing b
# MAGIC   ON d.product_id = b.material_id
# MAGIC WHERE b.material_id IS NULL;

# COMMAND ----------

# DBTITLE 1,Detailed product sales summary
# MAGIC %sql
# MAGIC -- Detailed view: All products with their sales metrics
# MAGIC SELECT 
# MAGIC   d.product_id,
# MAGIC   d.material_code_raw,
# MAGIC   d.product_description,
# MAGIC   d.brand,
# MAGIC   d.category,
# MAGIC   COUNT(b.billing_doc) as sales_transactions,
# MAGIC   COALESCE(SUM(b.quantity), 0) as total_quantity_sold,
# MAGIC   COALESCE(ROUND(SUM(b.net_value), 2), 0) as total_sales_value
# MAGIC FROM workspace.cbl_silver.dim_product d
# MAGIC LEFT JOIN workspace.cbl_silver.fact_billing b
# MAGIC   ON d.product_id = b.material_id
# MAGIC GROUP BY d.product_id, d.material_code_raw, d.product_description, d.brand, d.category
# MAGIC ORDER BY total_sales_value DESC;

# COMMAND ----------

# DBTITLE 1,Compare MATNR in bronze vs silver
# MAGIC %sql
# MAGIC -- 1. Show all MATNR values from bronze mara_material
# MAGIC SELECT 
# MAGIC   MATNR as bronze_matnr,
# MAGIC   MAKTX as material_description,
# MAGIC   ZZBRAND as brand
# MAGIC FROM workspace.cbl_bronze.mara_material
# MAGIC ORDER BY CAST(MATNR AS INT);

# COMMAND ----------

# DBTITLE 1,Show distinct material_id in fact_billing
# MAGIC %sql
# MAGIC -- 2. Show distinct material_id values from silver fact_billing
# MAGIC SELECT 
# MAGIC   material_id as silver_material_id,
# MAGIC   COUNT(*) as transaction_count
# MAGIC FROM workspace.cbl_silver.fact_billing
# MAGIC GROUP BY material_id
# MAGIC ORDER BY CAST(silver_material_id AS DECIMAL(18,0));

# COMMAND ----------

# DBTITLE 1,Find materials with no sales
# MAGIC %sql
# MAGIC -- 3. Find materials from bronze that have NO matching sales in fact_billing
# MAGIC -- Using raw MATNR (not padded) to join
# MAGIC WITH bronze_materials AS (
# MAGIC   SELECT DISTINCT
# MAGIC     MATNR as bronze_matnr,
# MAGIC     MAKTX as material_description,
# MAGIC     ZZBRAND as brand
# MAGIC   FROM workspace.cbl_bronze.mara_material
# MAGIC ),
# MAGIC sales_materials AS (
# MAGIC   SELECT DISTINCT material_id
# MAGIC   FROM workspace.cbl_silver.fact_billing
# MAGIC )
# MAGIC SELECT 
# MAGIC   m.bronze_matnr,
# MAGIC   m.material_description,
# MAGIC   m.brand,
# MAGIC   CASE 
# MAGIC     WHEN s.material_id IS NOT NULL THEN 'HAS SALES'
# MAGIC     ELSE 'NO SALES'
# MAGIC   END as sales_status
# MAGIC FROM bronze_materials m
# MAGIC LEFT JOIN sales_materials s
# MAGIC   ON m.bronze_matnr = s.material_id  -- Direct comparison (no padding)
# MAGIC ORDER BY sales_status, CAST(m.bronze_matnr AS INT);

# COMMAND ----------

# DBTITLE 1,Summary: Materials without sales
# MAGIC %sql
# MAGIC -- 4. Summary count
# MAGIC WITH bronze_materials AS (
# MAGIC   SELECT DISTINCT MATNR as bronze_matnr
# MAGIC   FROM workspace.cbl_bronze.mara_material
# MAGIC ),
# MAGIC sales_materials AS (
# MAGIC   SELECT DISTINCT material_id
# MAGIC   FROM workspace.cbl_silver.fact_billing
# MAGIC )
# MAGIC SELECT 
# MAGIC   COUNT(m.bronze_matnr) as total_materials_in_bronze,
# MAGIC   COUNT(s.material_id) as materials_with_sales,
# MAGIC   COUNT(m.bronze_matnr) - COUNT(s.material_id) as materials_without_sales
# MAGIC FROM bronze_materials m
# MAGIC LEFT JOIN sales_materials s
# MAGIC   ON m.bronze_matnr = s.material_id;

# COMMAND ----------

# DBTITLE 1,Sample raw string values from NETWR
# MAGIC %sql
# MAGIC -- Show 5 distinct raw string values from NETWR (net value)
# MAGIC SELECT DISTINCT NETWR
# MAGIC FROM workspace.cbl_bronze.sap_billing
# MAGIC WHERE NETWR IS NOT NULL
# MAGIC LIMIT 5;

# COMMAND ----------

# DBTITLE 1,Fact billing summary metrics
# MAGIC %sql
# MAGIC -- Summary metrics for fact_billing (F2 invoices)
# MAGIC SELECT
# MAGIC   COUNT(*) as row_count,
# MAGIC   COUNT(DISTINCT billing_doc) as distinct_billing_docs,
# MAGIC   ROUND(SUM(quantity), 3) as total_units,
# MAGIC   ROUND(SUM(net_value), 2) as total_net_value
# MAGIC FROM workspace.cbl_silver.fact_billing;

# COMMAND ----------

# DBTITLE 1,Build dim_distributor from distributor_master_raw
from pyspark.sql.functions import col, trim, regexp_replace, to_date, current_timestamp, lit, row_number
from pyspark.sql.window import Window
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_bronze}.distributor_master_raw"
target_table = f"{catalog}.{schema_silver}.dim_distributor"

print(f"Building dimension table: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read bronze distributor raw table
df_dist_raw = spark.table(source_table)

print(f"Total rows in bronze (including headers/footers): {df_dist_raw.count():,}\n")

# Add row number to identify position
window_spec = Window.orderBy(lit(1))
df_with_row = df_dist_raw.withColumn("row_num", row_number().over(window_spec))

# Filter to keep only data rows (skip first 5 rows which are titles/blanks/header)
df_data = df_with_row.filter(col("row_num") > 5)

# Get header from row 5 (index 4 in 0-based, but row_num 5 in 1-based)
header_row = df_with_row.filter(col("row_num") == 5).first()
headers = [
    trim(lit(header_row[f"col_0{i}"]) if header_row[f"col_0{i}"] else f"col_0{i}").alias(f"col_0{i}") 
    for i in range(7)
]

# Strip trailing spaces from header values and create proper column names
header_names = [
    header_row[f"col_0{i}"].strip() if header_row[f"col_0{i}"] else f"col_0{i}" 
    for i in range(7)
]

print(f"Header row (row 5): {header_names}\n")

# Rename columns based on header
df_renamed = df_data.select(
    col("col_00").alias("distributor_code"),
    col("col_01").alias("region"),
    col("col_02").alias("fleet_size_raw"),
    col("col_03").alias("service_level_raw"),
    col("col_04").alias("contract_signed_raw"),
    col("col_05").alias("notes"),
    col("_source_file"),
    col("_ingest_timestamp")
)

# Filter out blank rows, TOTAL rows, and footer rows
df_filtered = (df_renamed
    .filter(col("distributor_code").isNotNull())
    .filter(~(col("distributor_code").startswith("TOTAL")))
    .filter(~(col("distributor_code").startswith("Source:")))
    .filter(col("distributor_code") != "")
)

print(f"Rows after filtering blanks/totals/footers: {df_filtered.count():,}\n")

# Transform data types
df_transformed = (df_filtered
    # Fleet size: cast to integer
    .withColumn("fleet_size", col("fleet_size_raw").cast("int"))
    
    # Service Level %: remove %, replace comma with dot, convert to decimal
    # Format: "97,80000305175781%" -> 0.9780
    .withColumn("service_level_pct", 
                (regexp_replace(regexp_replace(col("service_level_raw"), "%", ""), ",", ".").cast("double") / 100).cast("decimal(5,4)")
               )
    
    # Contract Signed: convert dd/MM/yyyy to date
    .withColumn("contract_signed_date", to_date(col("contract_signed_raw"), "dd/MM/yyyy"))
    
    # Clean string columns
    .withColumn("distributor_code", trim(col("distributor_code")))
    .withColumn("region", trim(col("region")))
    .withColumn("notes", trim(col("notes")))
    
    # Add metadata
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
    
    # Select final columns
    .select(
        "distributor_code",
        "region",
        "fleet_size",
        "service_level_pct",
        "contract_signed_date",
        "notes",
        "_source_file",
        "_ingest_timestamp",
        "_transform_timestamp",
        "_batch_id"
    )
)

print("Sample transformed data:")
display(df_transformed.select("distributor_code", "region", "fleet_size", "service_level_pct", "contract_signed_date", "notes").limit(5))

# Write to silver with overwrite
df_transformed.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

row_count = df_transformed.count()
print(f"\n✓ Dimension table created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {row_count:,}")

# COMMAND ----------

# DBTITLE 1,Verify dim_distributor table
# MAGIC %sql
# MAGIC -- Verify the dim_distributor table
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_distributors,
# MAGIC   COUNT(DISTINCT region) as unique_regions,
# MAGIC   SUM(fleet_size) as total_fleet_size,
# MAGIC   ROUND(AVG(service_level_pct), 4) as avg_service_level,
# MAGIC   MIN(contract_signed_date) as earliest_contract,
# MAGIC   MAX(contract_signed_date) as latest_contract
# MAGIC FROM workspace.cbl_silver.dim_distributor;
# MAGIC
# MAGIC -- Show all distributors
# MAGIC SELECT *
# MAGIC FROM workspace.cbl_silver.dim_distributor
# MAGIC ORDER BY distributor_code;

# COMMAND ----------

# DBTITLE 1,Summary statistics for dim_distributor
# MAGIC %sql
# MAGIC -- Summary statistics for dim_distributor
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_distributors,
# MAGIC   COUNT(DISTINCT region) as unique_regions,
# MAGIC   SUM(fleet_size) as total_fleet_size,
# MAGIC   ROUND(AVG(service_level_pct), 4) as avg_service_level,
# MAGIC   ROUND(MIN(service_level_pct), 4) as min_service_level,
# MAGIC   ROUND(MAX(service_level_pct), 4) as max_service_level,
# MAGIC   MIN(contract_signed_date) as earliest_contract,
# MAGIC   MAX(contract_signed_date) as latest_contract
# MAGIC FROM workspace.cbl_silver.dim_distributor;

# COMMAND ----------

# DBTITLE 1,Show distinct values in holiday calendar boolean columns
# MAGIC %sql
# MAGIC -- Show distinct values for each boolean column
# MAGIC SELECT 'public_holiday' as column_name, public_holiday as value, COUNT(*) as count
# MAGIC FROM workspace.cbl_bronze.holiday_calendar
# MAGIC GROUP BY public_holiday
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 'poya_day' as column_name, poya_day as value, COUNT(*) as count
# MAGIC FROM workspace.cbl_bronze.holiday_calendar
# MAGIC GROUP BY poya_day
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 'liquor_sales_prohibited' as column_name, liquor_sales_prohibited as value, COUNT(*) as count
# MAGIC FROM workspace.cbl_bronze.holiday_calendar
# MAGIC GROUP BY liquor_sales_prohibited
# MAGIC
# MAGIC ORDER BY column_name, value;

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_gold.dim_date from holiday_calendar
from pyspark.sql.functions import col, trim, upper, when, coalesce, to_date, current_timestamp, lit, expr
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_bronze}.holiday_calendar"
target_table = f"{catalog}.cbl_gold.dim_date"

print(f"Building gold dimension table: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read bronze holiday calendar
df_holiday = spark.table(source_table)

print(f"Bronze holiday calendar rows: {df_holiday.count():,}\n")

# Function to convert text variants to boolean
# Handles: yes, no, Y, N, TRUE, FALSE, 1, 0
def convert_to_boolean(column):
    return when(
        upper(trim(col(column))).isin('YES', 'Y', 'TRUE', '1'), True
    ).when(
        upper(trim(col(column))).isin('NO', 'N', 'FALSE', '0'), False
    ).otherwise(None)

# Transform to gold dimension
df_dim_date = (df_holiday
    # Parse cal_date with two formats: yyyy-MM-dd and dd/MM/yyyy
    .withColumn("calendar_date", 
                coalesce(
                    expr("try_to_date(cal_date, 'yyyy-MM-dd')"),
                    expr("try_to_date(cal_date, 'dd/MM/yyyy')")
                ))
    
    # Convert all text variants to proper booleans
    .withColumn("is_public_holiday", convert_to_boolean("public_holiday"))
    .withColumn("is_poya_day", convert_to_boolean("poya_day"))
    .withColumn("is_dry_day", convert_to_boolean("liquor_sales_prohibited"))
    
    # Add metadata
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
    
    # Select final columns
    .select(
        "calendar_date",
        "is_public_holiday",
        "is_poya_day",
        "is_dry_day",
        "_transform_timestamp",
        "_batch_id"
    )
)

print("Sample transformed data:")
display(df_dim_date.limit(10))

# Create gold schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.cbl_gold")

# Write to gold with overwrite
df_dim_date.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

row_count = df_dim_date.count()
print(f"\n✓ Gold dimension table created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {row_count:,}")

# COMMAND ----------

# DBTITLE 1,Count dry days and poya days
# MAGIC %sql
# MAGIC -- Count dry days and poya days
# MAGIC SELECT 
# MAGIC   SUM(CASE WHEN is_dry_day = TRUE THEN 1 ELSE 0 END) as dry_day_count,
# MAGIC   SUM(CASE WHEN is_poya_day = TRUE THEN 1 ELSE 0 END) as poya_day_count,
# MAGIC   COUNT(*) as total_days
# MAGIC FROM workspace.cbl_gold.dim_date;

# COMMAND ----------

# DBTITLE 1,Show dry days that are not poya days
# MAGIC %sql
# MAGIC -- Dry days that are NOT poya days
# MAGIC SELECT 
# MAGIC   calendar_date,
# MAGIC   is_public_holiday,
# MAGIC   is_poya_day,
# MAGIC   is_dry_day
# MAGIC FROM workspace.cbl_gold.dim_date
# MAGIC WHERE is_dry_day = TRUE 
# MAGIC   AND is_poya_day = FALSE
# MAGIC ORDER BY calendar_date;

# COMMAND ----------

# DBTITLE 1,Check SFA orders for duplicates and pending status
# MAGIC %sql
# MAGIC -- Data quality summary for SFA orders
# MAGIC WITH duplicate_check AS (
# MAGIC   SELECT 
# MAGIC     COUNT(DISTINCT orderId) as unique_orders,
# MAGIC     COUNT(*) as total_records,
# MAGIC     COUNT(*) - COUNT(DISTINCT orderId) as duplicate_count
# MAGIC   FROM workspace.cbl_bronze.sfa_orders
# MAGIC ),
# MAGIC sync_status_check AS (
# MAGIC   SELECT 
# MAGIC     COUNT(CASE WHEN syncStatus = 'SYNCED' THEN 1 END) as synced_count,
# MAGIC     COUNT(CASE WHEN syncStatus = 'PENDING' THEN 1 END) as pending_count,
# MAGIC     COUNT(CASE WHEN syncStatus NOT IN ('SYNCED', 'PENDING') THEN 1 END) as other_status_count
# MAGIC   FROM workspace.cbl_bronze.sfa_orders
# MAGIC ),
# MAGIC timezone_check AS (
# MAGIC   SELECT 
# MAGIC     COUNT(CASE WHEN capturedAt RLIKE '.*[+-]\\d{2}:\\d{2}$' THEN 1 END) as has_tz_offset,
# MAGIC     COUNT(CASE WHEN capturedAt NOT RLIKE '.*[+-]\\d{2}:\\d{2}$' THEN 1 END) as missing_tz_offset
# MAGIC   FROM workspace.cbl_bronze.sfa_orders
# MAGIC )
# MAGIC SELECT 
# MAGIC   d.unique_orders,
# MAGIC   d.total_records,
# MAGIC   d.duplicate_count,
# MAGIC   s.synced_count,
# MAGIC   s.pending_count,
# MAGIC   s.other_status_count,
# MAGIC   t.has_tz_offset,
# MAGIC   t.missing_tz_offset
# MAGIC FROM duplicate_check d
# MAGIC CROSS JOIN sync_status_check s
# MAGIC CROSS JOIN timezone_check t;

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_silver.sfa_orders_clean
from pyspark.sql.functions import col, row_number, when, regexp_replace, to_timestamp, current_timestamp, lit, concat
from pyspark.sql.window import Window
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_bronze}.sfa_orders"
target_table = f"{catalog}.{schema_silver}.sfa_orders_clean"

print(f"Building clean SFA orders table: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read bronze SFA orders
df_sfa_bronze = spark.table(source_table)

print(f"Bronze SFA orders: {df_sfa_bronze.count():,} rows\n")

# Step 1: Filter out PENDING status (incomplete orders)
df_synced_only = df_sfa_bronze.filter(col("syncStatus") != "PENDING")

print(f"After excluding PENDING: {df_synced_only.count():,} rows\n")

# Step 2: Deduplicate - keep only the latest syncedAt for each orderId
# Define window partitioned by orderId, ordered by syncedAt descending
window_spec = Window.partitionBy("orderId").orderBy(col("syncedAt").desc())

# Add row number and keep only rank 1 (latest)
df_deduplicated = (df_synced_only
    .withColumn("row_rank", row_number().over(window_spec))
    .filter(col("row_rank") == 1)
    .drop("row_rank")
)

print(f"After deduplication (latest syncedAt per orderId): {df_deduplicated.count():,} rows\n")

# Step 3: Handle capturedAt timezone
# Flag records where timezone offset is missing (doesn't end with +HH:MM or -HH:MM pattern)
df_with_tz_flag = (df_deduplicated
    .withColumn("_tz_was_assumed", 
                when(~col("capturedAt").rlike(".*[+-]\\d{2}:\\d{2}$"), True)
                .otherwise(False))
    
    # Add timezone offset +05:30 if missing, assuming Asia/Colombo
    .withColumn("capturedAt_fixed",
                when(col("_tz_was_assumed"), concat(col("capturedAt"), lit("+05:30")))
                .otherwise(col("capturedAt")))
)

# Step 4: Convert timestamps and add metadata
df_clean = (df_with_tz_flag
    .withColumn("captured_timestamp", to_timestamp(col("capturedAt_fixed")))
    .withColumn("synced_timestamp", to_timestamp(col("syncedAt")))
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
    .drop("capturedAt_fixed")  # Remove intermediate column
    .select(
        "orderId",
        "sapBillingDoc",
        "deviceId",
        "repId",
        "outlet",
        "lines",
        "orderChannel",
        "paymentTerms",
        "syncStatus",
        col("capturedAt").alias("captured_at_raw"),
        "captured_timestamp",
        col("syncedAt").alias("synced_at_raw"),
        "synced_timestamp",
        "_tz_was_assumed",
        "appVersion",
        "ingest_date",
        "_source_file",
        "_ingest_timestamp",
        "_transform_timestamp",
        "_batch_id"
    )
)

print("Sample transformed data:")
display(df_clean.select(
    "orderId", "repId", "captured_at_raw", "captured_timestamp", 
    "_tz_was_assumed", "synced_timestamp"
).limit(10))

# Write to silver with overwrite
df_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

final_count = df_clean.count()
tz_assumed_count = df_clean.filter(col("_tz_was_assumed") == True).count()

print(f"\n✓ Clean SFA orders table created successfully!")
print(f"  Table: {target_table}")
print(f"  Final order count: {final_count:,}")
print(f"  Records where timezone was assumed: {tz_assumed_count:,}")

# COMMAND ----------

# DBTITLE 1,Verify SFA orders clean table
# MAGIC %sql
# MAGIC -- Summary of clean SFA orders
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_orders,
# MAGIC   COUNT(DISTINCT repId) as unique_reps,
# MAGIC   COUNT(CASE WHEN _tz_was_assumed = TRUE THEN 1 END) as tz_assumed_count,
# MAGIC   MIN(captured_timestamp) as earliest_order,
# MAGIC   MAX(captured_timestamp) as latest_order
# MAGIC FROM workspace.cbl_silver.sfa_orders_clean;
# MAGIC
# MAGIC -- Show sample records where timezone was assumed
# MAGIC SELECT 
# MAGIC   orderId,
# MAGIC   repId,
# MAGIC   captured_at_raw,
# MAGIC   captured_timestamp,
# MAGIC   _tz_was_assumed
# MAGIC FROM workspace.cbl_silver.sfa_orders_clean
# MAGIC WHERE _tz_was_assumed = TRUE
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Analyze sapBillingDoc in SFA orders
# MAGIC %sql
# MAGIC -- Check sapBillingDoc nulls and matches with fact_billing
# MAGIC WITH sfa_summary AS (
# MAGIC   SELECT 
# MAGIC     COUNT(*) as total_orders,
# MAGIC     COUNT(CASE WHEN sapBillingDoc IS NULL THEN 1 END) as null_billing_doc_count,
# MAGIC     COUNT(CASE WHEN sapBillingDoc IS NOT NULL THEN 1 END) as has_billing_doc_count
# MAGIC   FROM workspace.cbl_silver.sfa_orders_clean
# MAGIC ),
# MAGIC billing_matches AS (
# MAGIC   SELECT 
# MAGIC     COUNT(DISTINCT s.orderId) as matched_orders
# MAGIC   FROM workspace.cbl_silver.sfa_orders_clean s
# MAGIC   INNER JOIN workspace.cbl_silver.fact_billing f
# MAGIC     ON s.sapBillingDoc = f.billing_doc
# MAGIC   WHERE s.sapBillingDoc IS NOT NULL
# MAGIC )
# MAGIC SELECT 
# MAGIC   s.total_orders,
# MAGIC   s.null_billing_doc_count,
# MAGIC   s.has_billing_doc_count,
# MAGIC   b.matched_orders as billing_doc_matched_in_fact_billing,
# MAGIC   s.has_billing_doc_count - b.matched_orders as billing_doc_not_matched
# MAGIC FROM sfa_summary s
# MAGIC CROSS JOIN billing_matches b;

# COMMAND ----------

# DBTITLE 1,Check REGIO values in kna1_customer
# MAGIC %sql
# MAGIC -- Check distinct REGIO values for standardization mapping
# MAGIC SELECT 
# MAGIC   REGIO,
# MAGIC   COUNT(*) as count
# MAGIC FROM workspace.cbl_bronze.kna1_customer
# MAGIC WHERE LOEVM != 'X' OR LOEVM IS NULL
# MAGIC GROUP BY REGIO
# MAGIC ORDER BY REGIO;

# COMMAND ----------

# DBTITLE 1,Compare joins: fact_billing to raw kna1_customer vs dim_outlet
# MAGIC %sql
# MAGIC -- Join 1: fact_billing to RAW bronze kna1_customer (with duplicates)
# MAGIC WITH bronze_join AS (
# MAGIC   SELECT COUNT(*) as row_count
# MAGIC   FROM workspace.cbl_silver.fact_billing f
# MAGIC   INNER JOIN workspace.cbl_bronze.kna1_customer c
# MAGIC     ON f.customer_id = c.KUNNR
# MAGIC ),
# MAGIC -- Join 2: fact_billing to CLEAN gold dim_outlet (deduplicated)
# MAGIC gold_join AS (
# MAGIC   SELECT COUNT(*) as row_count
# MAGIC   FROM workspace.cbl_silver.fact_billing f
# MAGIC   INNER JOIN workspace.cbl_gold.dim_outlet o
# MAGIC     ON f.customer_id = o.outlet_id
# MAGIC )
# MAGIC SELECT 
# MAGIC   b.row_count as bronze_kna1_join_count,
# MAGIC   g.row_count as gold_dim_outlet_join_count,
# MAGIC   b.row_count - g.row_count as difference
# MAGIC FROM bronze_join b
# MAGIC CROSS JOIN gold_join g;

# COMMAND ----------

# DBTITLE 1,Find orphaned customers: fact_billing without dim_outlet match
# MAGIC %sql
# MAGIC -- Find customer_ids in fact_billing that do NOT exist in dim_outlet
# MAGIC WITH orphaned_customers AS (
# MAGIC   SELECT 
# MAGIC     f.customer_id,
# MAGIC     COUNT(*) as billing_row_count
# MAGIC   FROM workspace.cbl_silver.fact_billing f
# MAGIC   LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC     ON f.customer_id = o.outlet_id
# MAGIC   WHERE o.outlet_id IS NULL
# MAGIC   GROUP BY f.customer_id
# MAGIC )
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_orphaned_billing_rows,
# MAGIC   COUNT(DISTINCT customer_id) as unique_orphaned_customers
# MAGIC FROM (
# MAGIC   SELECT f.customer_id
# MAGIC   FROM workspace.cbl_silver.fact_billing f
# MAGIC   LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC     ON f.customer_id = o.outlet_id
# MAGIC   WHERE o.outlet_id IS NULL
# MAGIC );
# MAGIC
# MAGIC -- Show 10 example orphaned customer_ids with their billing counts
# MAGIC SELECT 
# MAGIC   f.customer_id,
# MAGIC   COUNT(*) as billing_row_count,
# MAGIC   SUM(f.net_value) as total_net_value
# MAGIC FROM workspace.cbl_silver.fact_billing f
# MAGIC LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC   ON f.customer_id = o.outlet_id
# MAGIC WHERE o.outlet_id IS NULL
# MAGIC GROUP BY f.customer_id
# MAGIC ORDER BY billing_row_count DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Count total orphaned billing rows
# MAGIC %sql
# MAGIC -- Total count of orphaned billing rows and unique customers
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_orphaned_billing_rows,
# MAGIC   COUNT(DISTINCT f.customer_id) as unique_orphaned_customers
# MAGIC FROM workspace.cbl_silver.fact_billing f
# MAGIC LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC   ON f.customer_id = o.outlet_id
# MAGIC WHERE o.outlet_id IS NULL;

# COMMAND ----------

# DBTITLE 1,Capture total net_value BEFORE quarantine handling
# MAGIC %sql
# MAGIC -- Capture total net_value before any changes
# MAGIC SELECT 
# MAGIC   SUM(net_value) as total_net_value_before,
# MAGIC   COUNT(*) as total_rows_before
# MAGIC FROM workspace.cbl_silver.fact_billing;

# COMMAND ----------

# DBTITLE 1,Insert UNKNOWN member into dim_outlet
# MAGIC %sql
# MAGIC -- Insert UNKNOWN outlet into dim_outlet for unresolved customer references
# MAGIC INSERT INTO workspace.cbl_gold.dim_outlet
# MAGIC VALUES (
# MAGIC   'UNKNOWN',                    -- outlet_id
# MAGIC   'Unknown/Unresolved Outlet',  -- outlet_name
# MAGIC   NULL,                         -- street_address
# MAGIC   'Unknown',                    -- city
# MAGIC   'Unknown',                    -- region
# MAGIC   NULL,                         -- customer_tier
# MAGIC   NULL,                         -- area_type
# MAGIC   NULL,                         -- account_group
# MAGIC   NULL,                         -- created_date
# MAGIC   NULL,                         -- credit_limit
# MAGIC   NULL,                         -- cooler_count
# MAGIC   NULL,                         -- is_exclusive
# MAGIC   NULL,                         -- latitude
# MAGIC   NULL,                         -- longitude
# MAGIC   NULL,                         -- is_active
# MAGIC   current_timestamp(),          -- _transform_timestamp
# MAGIC   'UNKNOWN_MEMBER_INSERT'       -- _batch_id
# MAGIC );
# MAGIC
# MAGIC -- Verify insert
# MAGIC SELECT * FROM workspace.cbl_gold.dim_outlet WHERE outlet_id = 'UNKNOWN';

# COMMAND ----------

# DBTITLE 1,Create quarantine table for unresolved customers
# MAGIC %sql
# MAGIC -- Create quarantine table with orphaned customer records
# MAGIC CREATE OR REPLACE TABLE workspace.cbl_silver.quarantine_unresolved_customer AS
# MAGIC SELECT 
# MAGIC   f.billing_doc,
# MAGIC   f.billing_date,
# MAGIC   f.customer_id,
# MAGIC   f.material_id,
# MAGIC   f.plant,
# MAGIC   f.distribution_channel,
# MAGIC   f.division,
# MAGIC   f.quantity,
# MAGIC   f.net_value,
# MAGIC   f.tax_amount,
# MAGIC   CASE 
# MAGIC     WHEN f.customer_id IS NULL THEN 'NULL_CUSTOMER_ID'
# MAGIC     WHEN f.customer_id = '0000000000' THEN 'PLACEHOLDER_CUSTOMER_ID'
# MAGIC     WHEN f.customer_id = '#' THEN 'INVALID_CUSTOMER_ID'
# MAGIC     ELSE 'CUSTOMER_NOT_IN_MASTER_DATA'
# MAGIC   END as reason,
# MAGIC   current_timestamp() as quarantine_timestamp
# MAGIC FROM workspace.cbl_silver.fact_billing f
# MAGIC LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC   ON f.customer_id = o.outlet_id
# MAGIC WHERE o.outlet_id IS NULL;
# MAGIC
# MAGIC -- Summary of quarantine table
# MAGIC SELECT 
# MAGIC   reason,
# MAGIC   COUNT(*) as row_count,
# MAGIC   COUNT(DISTINCT customer_id) as unique_customers,
# MAGIC   SUM(net_value) as total_net_value
# MAGIC FROM workspace.cbl_silver.quarantine_unresolved_customer
# MAGIC GROUP BY reason
# MAGIC ORDER BY row_count DESC;

# COMMAND ----------

# DBTITLE 1,Update fact_billing: set customer_id = 'UNKNOWN' for orphaned records
# MAGIC %sql
# MAGIC -- Update fact_billing to set customer_id = 'UNKNOWN' for orphaned records
# MAGIC -- Using CREATE OR REPLACE to substitute UNKNOWN for unresolved customers
# MAGIC CREATE OR REPLACE TABLE workspace.cbl_silver.fact_billing AS
# MAGIC SELECT 
# MAGIC   f.client,
# MAGIC   f.billing_doc,
# MAGIC   f.line_item,
# MAGIC   f.billing_type,
# MAGIC   f.billing_date,
# MAGIC   f.created_date,
# MAGIC   f.changed_date,
# MAGIC   -- Replace orphaned customer_ids with 'UNKNOWN'
# MAGIC   CASE 
# MAGIC     WHEN o.outlet_id IS NULL THEN 'UNKNOWN'
# MAGIC     ELSE f.customer_id
# MAGIC   END as customer_id,
# MAGIC   f.material_id,
# MAGIC   f.plant,
# MAGIC   f.distribution_channel,
# MAGIC   f.division,
# MAGIC   f.vendor_id,
# MAGIC   f.quantity,
# MAGIC   f.unit_of_measure,
# MAGIC   f.net_value,
# MAGIC   f.tax_amount,
# MAGIC   f.currency,
# MAGIC   f.unit_price,
# MAGIC   f.discount_pct,
# MAGIC   f.payment_terms,
# MAGIC   f.order_channel,
# MAGIC   f.loyalty_id,
# MAGIC   f.promo_code,
# MAGIC   f.has_data_quality_issues,
# MAGIC   f._ingest_timestamp,
# MAGIC   f._batch_id
# MAGIC FROM workspace.cbl_silver.fact_billing f
# MAGIC LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC   ON f.customer_id = o.outlet_id;
# MAGIC
# MAGIC -- Verify update count
# MAGIC SELECT 
# MAGIC   COUNT(*) as unknown_customer_count,
# MAGIC   SUM(net_value) as unknown_customer_net_value
# MAGIC FROM workspace.cbl_silver.fact_billing
# MAGIC WHERE customer_id = 'UNKNOWN';

# COMMAND ----------

# DBTITLE 1,Verify total net_value is unchanged
# MAGIC %sql
# MAGIC -- Verify total net_value AFTER quarantine handling
# MAGIC WITH after_totals AS (
# MAGIC   SELECT 
# MAGIC     SUM(net_value) as total_net_value_after,
# MAGIC     COUNT(*) as total_rows_after
# MAGIC   FROM workspace.cbl_silver.fact_billing
# MAGIC ),
# MAGIC before_totals AS (
# MAGIC   SELECT 
# MAGIC     2267907832.21 as total_net_value_before,
# MAGIC     241096 as total_rows_before
# MAGIC )
# MAGIC SELECT 
# MAGIC   b.total_net_value_before,
# MAGIC   a.total_net_value_after,
# MAGIC   a.total_net_value_after - b.total_net_value_before as net_value_difference,
# MAGIC   b.total_rows_before,
# MAGIC   a.total_rows_after,
# MAGIC   a.total_rows_after - b.total_rows_before as row_difference,
# MAGIC   CASE 
# MAGIC     WHEN ABS(a.total_net_value_after - b.total_net_value_before) < 0.01 THEN '✓ PASS: Net value preserved'
# MAGIC     ELSE '✗ FAIL: Net value changed'
# MAGIC   END as validation_status
# MAGIC FROM before_totals b
# MAGIC CROSS JOIN after_totals a;

# COMMAND ----------

# DBTITLE 1,Verify 100% join coverage with UNKNOWN member
# MAGIC %sql
# MAGIC -- Verify all fact_billing rows now join to dim_outlet (including UNKNOWN)
# MAGIC WITH join_coverage AS (
# MAGIC   SELECT 
# MAGIC     COUNT(*) as total_fact_rows,
# MAGIC     SUM(CASE WHEN o.outlet_id IS NOT NULL THEN 1 ELSE 0 END) as matched_rows,
# MAGIC     SUM(CASE WHEN o.outlet_id IS NULL THEN 1 ELSE 0 END) as orphaned_rows
# MAGIC   FROM workspace.cbl_silver.fact_billing f
# MAGIC   LEFT JOIN workspace.cbl_gold.dim_outlet o
# MAGIC     ON f.customer_id = o.outlet_id
# MAGIC )
# MAGIC SELECT 
# MAGIC   total_fact_rows,
# MAGIC   matched_rows,
# MAGIC   orphaned_rows,
# MAGIC   ROUND(100.0 * matched_rows / total_fact_rows, 2) as join_coverage_pct,
# MAGIC   CASE 
# MAGIC     WHEN orphaned_rows = 0 THEN '✓ PASS: 100% join coverage achieved'
# MAGIC     ELSE '✗ FAIL: Still have orphaned rows'
# MAGIC   END as validation_status
# MAGIC FROM join_coverage;
# MAGIC
# MAGIC -- Show breakdown by customer type
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN o.outlet_id = 'UNKNOWN' THEN 'UNKNOWN (Unresolved)'
# MAGIC     ELSE 'Known Outlets'
# MAGIC   END as customer_type,
# MAGIC   COUNT(DISTINCT f.customer_id) as unique_customers,
# MAGIC   COUNT(*) as billing_rows,
# MAGIC   SUM(f.net_value) as total_net_value,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct_of_total_rows
# MAGIC FROM workspace.cbl_silver.fact_billing f
# MAGIC INNER JOIN workspace.cbl_gold.dim_outlet o
# MAGIC   ON f.customer_id = o.outlet_id
# MAGIC GROUP BY customer_type
# MAGIC ORDER BY billing_rows DESC;

# COMMAND ----------

# DBTITLE 1,Final validation: check for any remaining orphaned rows
# Final check: any remaining orphaned rows?
query = f"""
SELECT 
  COUNT(*) as remaining_orphaned_rows
FROM {catalog}.{schema_silver}.fact_billing f
LEFT JOIN {catalog}.{schema_gold}.dim_outlet o
  ON f.customer_id = o.outlet_id
WHERE o.outlet_id IS NULL
"""

result = spark.sql(query)
display(result)
print("\nIf zero, all rows are now covered!")

# COMMAND ----------

# DBTITLE 1,Assert silver layer row count
# CRITICAL VALIDATION: Row count must match expected value
EXPECTED_ROW_COUNT = 241096

query = f"""
SELECT COUNT(*) as row_count
FROM {catalog}.{schema_silver}.fact_billing
"""

actual_row_count = spark.sql(query).collect()[0]['row_count']

if actual_row_count != EXPECTED_ROW_COUNT:
    error_msg = (
        f"ASSERTION FAILED: Silver layer row count mismatch!\n"
        f"  Expected: {EXPECTED_ROW_COUNT:,} rows\n"
        f"  Actual:   {actual_row_count:,} rows\n"
        f"  Difference: {actual_row_count - EXPECTED_ROW_COUNT:+,} rows\n\n"
        f"This indicates a data quality issue. Investigation required before proceeding."
    )
    raise AssertionError(error_msg)

print(f"✓ ASSERTION PASSED: Row count is exactly {actual_row_count:,} as expected")

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_gold.dim_outlet from kna1_customer
from pyspark.sql.functions import col, row_number, when, trim, upper, current_timestamp, lit
from pyspark.sql.window import Window
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_silver}.kna1_customer"
target_table = f"{catalog}.{schema_gold}.dim_outlet"

print(f"Building gold outlet dimension: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read silver customer table
df_customer = spark.table(source_table)

print(f"Silver customer rows: {df_customer.count():,}\n")

# Step 1: Filter out deleted customers (LOEVM = 'X')
df_active = df_customer.filter((col("deletion_flag") != "X") | col("deletion_flag").isNull())

print(f"After excluding deleted (LOEVM='X'): {df_active.count():,} rows\n")

# Step 2: Standardize REGIO values
# Mapping inconsistent region names to standardized values
df_standardized = df_active.withColumn("region_standardized",
    when(trim(upper(col("region"))).isin("N CENTRAL", "NORTH CENTRAL", "NORTH-CENTRAL"), "North Central")
    .when(trim(upper(col("region"))).isin("N. WESTERN", "NORTH WESTERN", "NORTH-WESTERN", "NORTHWESTERN"), "North Western")
    .when(trim(upper(col("region"))).isin("SABARAGAMUVA", "SABARAGAMUWA"), "Sabaragamuwa")
    .when(trim(upper(col("region"))).isin("UVA", "UWA"), "Uva")
    .when(trim(upper(col("region"))).isin("WESTERN", "WESTERN PROVINCE", "WESTERN PROV."), "Western")
    .when(trim(upper(col("region"))) == "CENTRAL", "Central")
    .when(trim(upper(col("region"))) == "EASTERN", "Eastern")
    .when(trim(upper(col("region"))) == "NORTHERN", "Northern")
    .when(trim(upper(col("region"))) == "SOUTHERN", "Southern")
    .otherwise(trim(col("region")))  # Keep original if no match
)

# Step 3: Deduplicate - keep only latest ERDAT per KUNNR
# Define window partitioned by customer_id, ordered by created_date descending (NULL last)
window_spec = Window.partitionBy("customer_id").orderBy(col("created_date").desc_nulls_last())

# Add row number and keep only rank 1 (latest)
df_deduplicated = (df_standardized
    .withColumn("row_rank", row_number().over(window_spec))
    .filter(col("row_rank") == 1)
    .drop("row_rank")
)

print(f"After deduplication (latest ERDAT per KUNNR): {df_deduplicated.count():,} rows\n")

# Step 4: Build final dimension with metadata
df_dim_outlet = (df_deduplicated
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
    .select(
        col("customer_id").alias("outlet_id"),
        col("customer_name").alias("outlet_name"),
        "street_address",
        "city",
        col("region_standardized").alias("region"),
        "customer_tier",
        "area_type",
        "account_group",
        "created_date",
        "credit_limit",
        "cooler_count",
        "is_exclusive",
        "latitude",
        "longitude",
        "_transform_timestamp",
        "_batch_id"
    )
)

print("Sample outlet dimension data:")
display(df_dim_outlet.select(
    "outlet_id", "outlet_name", "city", "region", "customer_tier"
).limit(10))

# Create gold schema if it doesn't exist
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.cbl_gold")

# Write to gold with overwrite
df_dim_outlet.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

final_count = df_dim_outlet.count()
distinct_outlets = df_dim_outlet.select("outlet_id").distinct().count()

print(f"\n✓ Gold outlet dimension created successfully!")
print(f"  Table: {target_table}")
print(f"  Total rows written: {final_count:,}")
print(f"  Distinct outlets: {distinct_outlets:,}")

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_gold.fact_sales_daily aggregate
from pyspark.sql.functions import col, sum as _sum, count, countDistinct, current_timestamp, lit
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_silver}.fact_billing"
target_table = f"{catalog}.{schema_gold}.fact_sales_daily"

print(f"Building gold sales fact: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read silver fact_billing
df_billing = spark.table(source_table)

print(f"Silver fact_billing rows: {df_billing.count():,}\n")

# Aggregate to daily grain: billing_date x customer_id x material_id
df_sales_daily = (df_billing
    .groupBy(
        col("billing_date"),
        col("customer_id").alias("outlet_id"),
        col("material_id")
    )
    .agg(
        _sum("quantity").alias("total_units"),
        _sum("net_value").alias("total_net_value"),
        countDistinct("billing_doc").alias("distinct_billing_docs")
    )
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
)

print("Sample aggregated data:")
display(df_sales_daily.orderBy(col("billing_date").desc()).limit(10))

# Write to gold with overwrite
df_sales_daily.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

final_count = df_sales_daily.count()

print(f"\n✓ Gold sales fact created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {final_count:,}")

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_gold.dim_product from silver
from pyspark.sql.functions import col, current_timestamp, lit
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_silver}.dim_product"
target_table = f"{catalog}.{schema_gold}.dim_product"

print(f"Building gold product dimension: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read silver dim_product
df_product = spark.table(source_table)

print(f"Silver product rows: {df_product.count():,}\n")

# Add gold layer metadata
df_gold_product = (df_product
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
)

print("Sample product dimension data:")
display(df_gold_product.limit(10))

# Write to gold with overwrite
df_gold_product.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

final_count = df_gold_product.count()

print(f"\n✓ Gold product dimension created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {final_count:,}")

# COMMAND ----------

# DBTITLE 1,Build workspace.cbl_gold.dim_distributor from silver
from pyspark.sql.functions import col, current_timestamp, lit
import uuid

# Generate batch ID
batch_id = str(uuid.uuid4())

# Source and target
source_table = f"{catalog}.{schema_silver}.dim_distributor"
target_table = f"{catalog}.{schema_gold}.dim_distributor"

print(f"Building gold distributor dimension: {target_table}")
print(f"Source: {source_table}")
print(f"Batch ID: {batch_id}\n")

# Read silver dim_distributor
df_distributor = spark.table(source_table)

print(f"Silver distributor rows: {df_distributor.count():,}\n")

# Add gold layer metadata
df_gold_distributor = (df_distributor
    .withColumn("_transform_timestamp", current_timestamp())
    .withColumn("_batch_id", lit(batch_id))
)

print("Sample distributor dimension data:")
display(df_gold_distributor.limit(10))

# Write to gold with overwrite
df_gold_distributor.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

final_count = df_gold_distributor.count()

print(f"\n✓ Gold distributor dimension created successfully!")
print(f"  Table: {target_table}")
print(f"  Rows written: {final_count:,}")

# COMMAND ----------

# DBTITLE 1,Verify fact_sales_daily metrics
# Summary metrics for gold fact_sales_daily
query = f"""
SELECT 
  COUNT(*) as row_count,
  SUM(total_units) as total_units,
  SUM(total_net_value) as total_net_value
FROM {catalog}.{schema_gold}.fact_sales_daily
"""

result = spark.sql(query)
display(result)