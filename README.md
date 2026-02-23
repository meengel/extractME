# **extractME** *by Michael Engel*
This package provides the basic functionalities for extracting zonal statistics and empirical distributions of your regions of interest from (cloud-native) raster-data considering the true area of a pixel. It is designed on top of the [exactextract-package](https://isciences.github.io/exactextract/) incorporating a patched version of the [ERADist and ERANataf classes](https://github.com/ERA-Software/Overview/tree/main/ERADist/ERA_Distribution_Classes_Python/Classes) to support advanced statistical analysis ranging from Bayesian inference to data-aware optimization. It can be used to do your own analysis - simply credit ours by citing our [Zenodo DOI]()!

The package supports a variety of different zonal extraction procedures such as
- intersection based pixel weighting (standard case)
- fully contained pixel weighting
- partially contained pixel weighting
- all-touched pixel extraction
- boundary pixel extraction
- centroid based pixel extraction
- centroid based fully contained pixel extraction
- centroid based boundary pixel extraction

Based on these, it can be used for advanced analysis of zonal statistics using and comparing multiple approaches. One particular contribution of the package is to allow the correction of these pixel weighting approaches with respect to the true area covered by a pixel. Doing so, distortion effects as to the projection of the underlying rasterdata can be incorporated into the data extraction. Hence, no reprojection (and corresponding potentially erroneous resampling) is necessary anymore to acquire reasonable statistics. And maybe the best feature: it is designed for cloud-native geospatial workflows and supports direct reading from S3-compatible stores which makes it pretty efficient!

## Main Features
- extract zonal statistics in a single call at scale
- classic statistics like median, quantiles and means as well as empirical distribution objects
- raster input from local files or S3
- projection aware area weights
- exclude undesired values from the calculations
- returning a GeoDataFrame

## Authors (please use our [Zenodo DOI]()!)
- Michael Engel

## Installation
Currently, we do not provide the package via pip or conda. However, you may install it by installing `numpy`, `scipy`, `rasterio`, `geopandas`, `gdal`, `boto3`, `tqdm` and `exactextract`. In conda, this may look something like `conda create -n extractme -c conda-forge python numpy scipy matplotlib rasterio geopandas gdal boto3 tqdm exactextract`. If you want to store your raw-extractions or quantities to a parquet file, you need to install the respective dependencies for geopandas (i.e. `pyarrow`) as well.
Then, register the repository in your environment using `pip install -e /PATH/TO/extractME-repository` or similar.

## Usage Examples
There is an own [repository with examples](https://github.com/meengel/extractME-examples) for the advanced usage of the package. Nevertheless, we put some minimal examples here.

### Minimal
```python
from extractme import extractQuantity

polygonInput = "path/to/polygonFile.shp" # could also be some GeoDataFrame (e.g. if you want to filter before the query)
rasterInput = ["path/to/rasterFile1.jp2", "path/to/rasterFile2WithPotentiallyMultipleBands.tif"] # always a list of input paths to files with the same grid

result_gdf = extractQuantity(
    rasterInput,
    polygonInput,
    weightInput = "trueAreaForFirst", # calculates the true pixel area using LAEA projection of the first file using the same internal reading blocks if possible
    quantities = ["quantile-0p025", "median", "quantile-0p975"],
    centroidBasedQuantities = ["unweightedCentroidBasedMedian"],
)

print(result_gdf.head())
```

### Minimal With Weights
```python
from extractme import extractQuantity

polygonInput = "path/to/polygonFile.shp" # could also be some GeoDataFrame (e.g. if you want to filter before the query)
rasterInput = ["path/to/rasterFile1.jp2", "path/to/rasterFile2WithPotentiallyMultipleBands.tif"] # always a list of input paths to files with the same grid
weightInput = "path/to/weightFile.tif" # temperature, population density or similar with the same grid as rasterInput

result_gdf = extractQuantity(
    rasterInput,
    polygonInput,
    weightInput,
    quantities = ["quantile-0p025", "median", "quantile-0p975"],
    centroidBasedQuantities = ["unweightedCentroidBasedMedian"],
)

print(result_gdf.head())
```

### AWS S3 based Sentinel-2
```python
import datetime as dt
from extractme import extractQuantity

# 1. Polygon input
polygonInput = "path/to/polygonFile.shp" # could also be some GeoDataFrame (e.g. if you want to filter before the query)

# 2. Raster input: Build GDAL S3 paths
tile = "33UVT"
date = dt.datetime(year=2025, month=9, day=29)
swath = 0
BUCKET = "sentinel-s2-l1c"
base = f"/vsis3/{BUCKET}/tiles/" + "/".join([tile[:2], tile[2], tile[3:], str(date.year), str(date.month), str(date.day), str(swath)]) + "/"
bands_10m = ["B02.jp2", "B03.jp2", "B04.jp2", "B08.jp2"]

rasterInput = [base + b for b in bands_10m] # always a list of input paths to files with the same grid

# 3. Load AWS credentials (simple file-based example)
def load_aws_credentials(key_path, secret_path):
    with open(key_path) as k, open(secret_path) as s:
        return {"key": k.read().strip(), "secret": s.read().strip()}
creds = load_aws_credentials("KEY_FILE.txt", "SECRET_FILE.txt") # never ever insert the credentials hard coded!

# 4. Call extractQuantity
result_gdf = extractQuantity(
    rasterInput,
    polygonInput,
    weightInput = "trueAreaForFirst", # calculates the true pixel area using LAEA projection of the first file using the same internal reading blocks if possible

    quantities = ["median", "quantile-0p95", "std", {"empDigest":{"nApprox": 1000}}],
    centroidBasedQuantities = ["unweightedCentroidBasedMedian", "centroidBasedQuantile-0p95", "centroidBasedStd", {"centroidBasedEmpDist":{}}],
    ignoreValues = [0],

    pointsPerEdgeAreaCalculation = 2, # number of points of an edge starting from a point; has to be greater or equal than 1 (the larger, the more precise)!
    nWorkersAreaCalculation = 6, # number of workers used to calculate the pixel areas

    keepRawExtraction = False, # keep values, coverage and weights as returned by exactextract
    keepNonIntersecting = False, # keep polygons with no extraction results

    key = creds["key"],
    secret = creds["secret"],
    aws_region = "eu-central-1",

    progress = True, # print progress of computation
)

# 5. Analyse results
print(result_gdf.head())
```

## Column Naming Convention
Since it is possible to insert rasters with multiple bands, the package always labels the columns by the stem of the path first, then the band to be followed by the respective quantity. For example, if the raster input is `["path/to/File1With2Bands.tif", "path/to/File2with1Band.jp2"]` and you choose the quantities to be `["median", "quantile-0p025"]`, then the resulting column names are `["File1With2Bands_band_1_median", "File1With2Bands_band_1_quantile-0p025", "File1With2Bands_band_2_median", "File1With2Bands_band_2_quantile-0p025", "File2with1Band_band_1_median", "File2with1Band_band_1_quantile-0p025"]`.

## Available Quantities
In the following, there will be a detailed list, but in general, the package supports the following quantities:
- EmpDist
- EmpDigest
- Quantiles
- Median
- Mean
- Std
- Var
- Weight

### **EmpDist**
The `EmpDist` quantity refers to the corresponding initialization of the `ERADist` class implementing an empirical distribution. It is based on the inverted-cdf approach and, accordingly, allows the query of any quantile or other statistical property from it. Further, as to its implementation as part of `ERADist`, it supports the usage within isoprobabilistic transformations such as the Nataf-Transformation implemented by `ERANataf`. Please note that it uses all values und weights of a polygon of interest which may result in high memory consumption. It is possible to join multiple instances by simple summation whereas the resulting objects inherits the initialization parameters of the first summand (we highly recommend using parallelized reducing methods for this step!).

It can be initialized by inserting a dictionary with the respective quantity as a key and the initialization keyword arguments as respective values:
- `pdfMethod`: Desired method for the PDF creation. The default is nearest.
- `pdfPoints`: Desired number of points for the PDF creation. The default is None (will resolve in the square root of the number of data points).
- `pdfMethodParams`: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.

#### intersection based
- `empDist`
- `allTouchedEmpDist`
- `boundaryEmpDist`
- `fullyContainedEmpDist`
- `partiallyContainedEmpDist`
- `unweightedEmpDist`
- `unweightedAllTouchedEmpDist`
- `unweightedBoundaryEmpDist`
- `unweightedFullyContainedEmpDist`
- `unweightedPartiallyContainedEmpDist`

#### centroid based
- `centroidBasedEmpDist`
- `centroidBasedBoundaryEmpDist`
- `centroidBasedFullyContainedEmpDist`
- `unweightedCentroidBasedEmpDist`
- `unweightedCentroidBasedBoundaryEmpDist`
- `unweightedCentroidBasedFullyContainedEmpDist`

### **EmpDigest** (experimental feature)
The `EmpDigest` quantity refers to the corresponding initialization of the `ERADist` class implementing a compressed version of the empirical distribution. It is similar to approaches like the [t-digest](https://github.com/tdunning/t-digest) whereas, here, the number of values and weights is fixed. That fixed number of samples are chosen as the most representative for a desired amount of probability mass. So in the end, it is just a reduced/approximate form of an empirical distribution. That is, it also allows the query of any quantile or other statistical property. Further, as to its implementation as part of `ERADist`, it supports the usage within isoprobabilistic transformations such as the Nataf-Transformation implemented by `ERANataf`. Please note that it solely uses a desired number of values und weights of a polygon of interest. It is possible to join multiple instances by simple summation whereas the resulting objects inherits the initialization parameters of the first summand (we highly recommend using pairwise parallelized reducing methods for this step since the merging order actually matters a lot!).

It can be initialized by inserting a dictionary with the respective quantity as a key and the initialization keyword arguments as respective values:
- `nApprox`: Desired number of points to approximate the empirical CDF in standard normal space. The default is 1000.
- `mode`: Desired mode to construct the approximation points in standard normal space. The default is 'densityFocused'.
- `eps`: Desired infinitesimal to check the maximum distinguishable values in standard normal space. The default is None.
- `pdfMethod`: Desired method for the PDF creation. The default is nearest.
- `pdfPoints`: Desired number of points for the PDF creation. The default is None (will resolve in the square root of the number of data points).
- `pdfMethodParams`: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.

#### intersection based
- `empDigest`
- `allTouchedEmpDigest`
- `boundaryEmpDigest`
- `fullyContainedEmpDigest`
- `partiallyContainedEmpDigest`
- `unweightedEmpDigest`
- `unweightedAllTouchedEmpDigest`
- `unweightedBoundaryEmpDigest`
- `unweightedFullyContainedEmpDigest`
- `unweightedPartiallyContainedEmpDigest`

#### centroid based
- `centroidBasedEmpDigest`
- `centroidBasedBoundaryEmpDigest`
- `centroidBasedFullyContainedEmpDigest`
- `unweightedCentroidBasedEmpDigest`
- `unweightedCentroidBasedBoundaryEmpDigest`
- `unweightedCentroidBasedFullyContainedEmpDigest`

### **Quantiles**
The quantile quantities denote the quantiles as defined by the patterned name: e.g. "quantile-0p025" queries the 2.5% quantile. They are calculated using the inverted-cdf method.
#### intersection based
- `quantile-0pXX...`
- `allTouchedQuantile-0pXX...`
- `boundaryQuantile-0pXX...`
- `fullyContainedQuantile-0pXX...`
- `partiallyContainedQuantile-0pXX...`
- `unweightedQuantile-0pXX...`
- `unweightedAllTouchedQuantile-0pXX...`
- `unweightedBoundaryQuantile-0pXX...`
- `unweightedFullyContainedQuantile-0pXX...`
- `unweightedPartiallyContainedQuantile-0pXX...`

#### centroid based
- `centroidBasedQuantile-0pXX...`
- `centroidBasedBoundaryQuantile-0pXX...`
- `centroidBasedFullyContainedQuantile-0pXX...`
- `unweightedCentroidBasedQuantile-0pXX...`
- `unweightedCentroidBasedBoundaryQuantile-0pXX...`
- `unweightedCentroidBasedFullyContainedQuantile-0pXX...`

### **Median**
The median value is calculated using the inverted-cdf method. Actually, it could also be queried using the quantile quantity setting `0p5`.
#### intersection based
- `median`
- `allTouchedMedian`
- `boundaryMedian`
- `fullyContainedMedian`
- `partiallyContainedMedian`
- `unweightedMedian`
- `unweightedAllTouchedMedian`
- `unweightedBoundaryMedian`
- `unweightedFullyContainedMedian`
- `unweightedPartiallyContainedMedian`

#### centroid based
- `centroidBasedMedian`
- `centroidBasedBoundaryMedian`
- `centroidBasedFullyContainedMedian`
- `unweightedCentroidBasedMedian`
- `unweightedCentroidBasedBoundaryMedian`
- `unweightedCentroidBasedFullyContainedMedian`

### **Mean**
The mean value is calculated using the unbiased sample mean.
#### intersection based
- `mean`
- `allTouchedMean`
- `boundaryMean`
- `fullyContainedMean`
- `partiallyContainedMean`
- `unweightedMean`
- `unweightedAllTouchedMean`
- `unweightedBoundaryMean`
- `unweightedFullyContainedMean`
- `unweightedPartiallyContainedMean`

#### centroid based
- `centroidBasedMean`
- `centroidBasedBoundaryMean`
- `centroidBasedFullyContainedMean`
- `unweightedCentroidBasedMean`
- `unweightedCentroidBasedBoundaryMean`
- `unweightedCentroidBasedFullyContainedMean`

### **Std**
The standard deviation is the square root of the unbiased sample variance.
#### intersection based
- `std`
- `allTouchedStd`
- `boundaryStd`
- `fullyContainedStd`
- `partiallyContainedStd`
- `unweightedStd`
- `unweightedAllTouchedStd`
- `unweightedBoundaryStd`
- `unweightedFullyContainedStd`
- `unweightedPartiallyContainedStd`

#### centroid based
- `centroidBasedStd`
- `centroidBasedBoundaryStd`
- `centroidBasedFullyContainedStd`
- `unweightedCentroidBasedStd`
- `unweightedCentroidBasedBoundaryStd`
- `unweightedCentroidBasedFullyContainedStd`

### **Var**
The variance is calculated using the unbiased sample variance.
#### intersection based
- `var`
- `allTouchedVar`
- `boundaryVar`
- `fullyContainedVar`
- `partiallyContainedVar`
- `unweightedVar`
- `unweightedAllTouchedVar`
- `unweightedBoundaryVar`
- `unweightedFullyContainedVar`
- `unweightedPartiallyContainedVar`

#### centroid based
- `centroidBasedVar`
- `centroidBasedBoundaryVar`
- `centroidBasedFullyContainedVar`
- `unweightedCentroidBasedVar`
- `unweightedCentroidBasedBoundaryVar`
- `unweightedCentroidBasedFullyContainedVar`

### **Weight**
The weight quantity denotes the sum of all weights. For example, if the `weightInput` parameter is equal to `"trueAreaForFirst"`, then the weight denotes the true area of the respective polygon. If an unweighted weight is desired, the weight denotes the number of pixels. To get the nominal area as defined by the respective crs of the input raster data, one has to multiply that value as to the nominal area per pixel.
#### intersection based
- `weight`
- `allTouchedWeight`
- `boundaryWeight`
- `fullyContainedWeight`
- `partiallyContainedWeight`
- `unweightedWeight`
- `unweightedAllTouchedWeight`
- `unweightedBoundaryWeight`
- `unweightedFullyContainedWeight`
- `unweightedPartiallyContainedWeight`

#### centroid based
- `centroidBasedWeight`
- `centroidBasedBoundaryWeight`
- `centroidBasedFullyContainedWeight`
- `unweightedCentroidBasedWeight`
- `unweightedCentroidBasedBoundaryWeight`
- `unweightedCentroidBasedFullyContainedWeight`
