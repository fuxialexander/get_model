# %%
import os
import pyspark
import numpy as np

os.environ['SPARK_HOME'] = '/manitou/pmg/users/xf2217/mambaforge/hail/lib/python3.11/site-packages/pyspark/'
# use ivy2 jars in spark

os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 200g --packages org.apache.hadoop:hadoop-aws:3.2.2 pyspark-shell'
# ivy2 cache and jar path

# set java heap size
os.environ['JAVA_OPTS'] = '-Xmx200g'
from pyspark.sql import SparkSession
HAIL_DIR = '/manitou/pmg/users/xf2217/mambaforge/hail/lib/python3.11/site-packages/hail/'
spark = SparkSession.builder \
    .appName("Hail") \
    .config('spark.jars', f'{HAIL_DIR}/backend/hail-all-spark.jar') \
    .config('spark.driver.extraClassPath', f'{HAIL_DIR}/backend/hail-all-spark.jar') \
    .config('spark.executor.extraClassPath', f'{HAIL_DIR}/backend/hail-all-spark.jar') \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator") \
    .config('spark.kryo.registrationRequired', 'false')\
    .config('spark.kryoserializer.buffer.max', '2047m')\
    .config('spark.executor.memory', '200g') \
    .config('spark.executor.num', '24') \
    .config('spark.driver.maxResultSize', '200g') \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")\
    .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider')\
    .getOrCreate()
sc = spark.sparkContext
#%%
import hail as hl
hl.init(sc=sc, local='local[96]')
hg37 = hl.get_reference('GRCh37')  
hg38 = hl.get_reference('GRCh38') 
ukbb_dir = '/pmglocal/xf2217/ukbb/'
hg37.add_liftover('/manitou/pmg/users/xf2217/get_model/data/grch37_to_grch38.over.chain.gz', hg38)
# copy ukbb data from central storage to local storage
# create a folder in local storage
pop = 'AFR'
os.makedirs('/pmglocal/xf2217/ukbb/', exist_ok=True)
os.system(f'cp -r /manitou/pmg/users/xf2217/ukbb/UKBB.{pop}.ldadj.variant.ht /pmglocal/xf2217/ukbb/')
variant_index_url = f'{ukbb_dir}/UKBB.{pop}.ldadj.variant.ht'
variant_index = hl.read_table(variant_index_url)
#%%
# %%
s3_url = f's3a://pan-ukb-us-east-1/ld_release/UKBB.{pop}.ldadj.bm'
bm = hl.linalg.BlockMatrix.read(s3_url)
#%%
# query a region chr1:1-1000000 for all variants
region = 'chr1:4000000-8000000'
anno = variant_index.filter((variant_index.locus.contig == region.split(':')[0]) & (variant_index.locus.position >= int(region.split(':')[1].split('-')[0])) & (variant_index.locus.position <= int(region.split(':')[1].split('-')[1])))
anno_idx = anno.idx.collect()
#%%
bm_result = bm[min(anno_idx):max(anno_idx)+1, min(anno_idx):max(anno_idx)+1].to_numpy()

import zarr
# save bm_result to zarr as array
zarr_path = '/pmglocal/xf2217/ukbb/ldadj.zarr'
bm_result[bm_result<0.8] = 0 # set the threshold to 0.1
# set the diagonal to 0
np.fill_diagonal(bm_result, 0)
zarr.convenience.save_array(zarr_path, bm_result, chunks=(1000,1000), compressor=zarr.Blosc(cname='zstd', clevel=1, shuffle=2)) 
# %%
s3_mt_url = 's3a://pan-ukb-us-east-1/sumstats_release/results_full.mt'
mt = hl.read_matrix_table(s3_mt_url)
# %%
mt.describe()
# %%
mt.count()