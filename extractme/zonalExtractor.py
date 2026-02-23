from pathlib import Path
from functools import partial
import rasterio
from rasterio.env import Env
import geopandas as gpd
from tqdm import tqdm
tqdm.pandas()

import numpy as np

from .libs import utils
from .libs import statUtils
from .libs import rasterioUtils
from .areaCalculator import calculatePixelAreaTif

import importlib
ee = importlib.import_module("exactextract.exact_extract")
ee.prep_raster = utils._prep_raster_patched # path raster preparation
exact_extract = ee.exact_extract

def extractChosenOps(
    rasterInput, # list of file paths (local or /vsis3/ or s3://)
    polygonInput, # local path or GeoDataFrame
    weightInput = None, # None, list of paths, or trueAreaForFirst or trueAreaForEach
    ops = ["values", "coverage", "weights", "center_x", "center_y"], # see https://isciences.github.io/exactextract/operations.html
    
    keepColumns=None,
    idColumn=None,
    idColumnFallbackName="id",
    savenameRawExtraction=None,
    savetypeRawExtraction="GPKG",
    savenameAreaCalculation=None, # None is handled by calculatePixelAreaTif
    blocksizeAreaCalculation=None, # None -> infers blocksize from source file; only change if you know what you are doing
    pointsPerEdgeAreaCalculation=1,
    nWorkersAreaCalculation=0,
    
    key=None,
    secret=None,
    s3_endpoint=None, # host-like value, e.g. "s3.eu-central-1.amazonaws.com" or "my-s3.example.com"
    requester_pays=False,
    aws_region=None,
    enforceS3Session=False,
    allowPythonVrt=False,
    
    progress = True,
    **exactextract_kwargs
):
    # Prepare vector input
    if isinstance(polygonInput, gpd.GeoDataFrame):
        gdfFull = polygonInput
    else:
        gdfFull = gpd.read_file(polygonInput)

    # Normalize weights
    deleteAreaResult = False
    if weightInput:
        if weightInput == "trueAreaForFirst":
            if progress:
                print("Starting computation of pixel areas for the first raster file only!")
            weights = calculatePixelAreaTif(
                src_path = rasterInput[0],
                dst_path = savenameAreaCalculation,
                points_per_edge = pointsPerEdgeAreaCalculation,
                block_size = blocksizeAreaCalculation,
                num_workers = nWorkersAreaCalculation,
                nodata = np.nan,
                fallbackBlocksize = 512,
                progress = progress,
                key = key,
                secret = secret,
                s3_endpoint = s3_endpoint,
                requester_pays = requester_pays,
                aws_region = aws_region,
            )
            deleteAreaResult = True if savenameAreaCalculation is None else False
        elif weightInput == "trueAreaForEach":
            if progress:
                print("Starting computation of pixel areas for the each raster file!")
            weights = [
                calculatePixelAreaTif(
                    src_path = rasterInput[i],
                    dst_path = utils._lenList(savenameAreaCalculation, len(rasterInput))[i],
                    points_per_edge = pointsPerEdgeAreaCalculation,
                    block_size = blocksizeAreaCalculation,
                    num_workers = nWorkersAreaCalculation,
                    nodata = np.nan,
                    fallbackBlocksize = 512,
                    progress = progress,
                    key = key,
                    secret = secret,
                    s3_endpoint = s3_endpoint,
                    requester_pays = requester_pays,
                    aws_region = aws_region,
                ) for i in tqdm(range(len(rasterInput)), desc="Pixel Area Calculation", disable=not progress)
            ]
            deleteAreaResult = True if savenameAreaCalculation is None else False
        else:
            weights = weightInput
    else:
        weights = None
    weights = utils._lenList(weights, len(rasterInput))

    if keepColumns is None:
        keepColumns = []
    if idColumn is None:
        gdfFull[idColumnFallbackName] = gdfFull.index
        idColumn = idColumnFallbackName
    keepColumns.append(idColumn)
    keepColumns = list(set(keepColumns))[::-1]

    # allow Python scripts in VRT files if desired
    gdalEnv = {}
    if allowPythonVrt:
        gdalEnv.update(GDAL_VRT_ENABLE_PYTHON="YES")
    
    # decide whether any S3 handling is required
    any_s3 = any(utils._is_s3_path(p) for p in list(rasterInput) + list(weights))
    if any_s3 or enforceS3Session:
        # Run the full sequence inside rasterio.Env + temporary env so GDAL/rasterio see the session and endpoint
        s3session, gdal_env = utils.getRasterioAwsSession(key, secret, s3_endpoint, requester_pays, aws_region)
        gdalEnv.update(gdal_env)
    else:
        s3session=None
        
    with Env(session=s3session, AWS_VIRTUAL_HOSTING=False):
        with utils._temporary_environ(gdalEnv):
            # determine raster CRS from first raster and reproject vector
            with rasterio.open(rasterInput[0]) as ref:
                raster_crs = ref.crs
                vec_crs = gdfFull.crs
            gdfFull = gdfFull.to_crs(raster_crs)
            
            if progress:
                print("Starting actual data extraction!")
            with rasterioUtils.openRasterListWithFallback(rasterInput) as rasterSources:
                with rasterioUtils.openRasterListWithFallback(weights) as weightSources:
                    resultGdf = exact_extract(
                        rast = rasterSources,
                        vec = gdfFull,
                        ops = ops,
                        weights = weightSources,
                        include_cols = keepColumns,
                        include_geom = True,
                        strategy = exactextract_kwargs.get("strategy", "feature-sequential"),
                        max_cells_in_memory = exactextract_kwargs.get("max_cells_in_memory", 30000000),
                        grid_compat_tol = exactextract_kwargs.get("grid_compat_tol", 0.001),
                        output = "pandas",
                        output_options = exactextract_kwargs.get("output_options", None),
                        progress = progress
                    )
            renamingPairs = utils._getRenamingPairs(rasterInput, operations=ops)
    
    if deleteAreaResult:
        if progress:
            print("Deleting computation result of pixel areas!")
        utils.delete(list(np.unique(weights)))
    
    if progress:
        print(f"Reprojecting resulting GeoDataFrame from raster crs {raster_crs} to the original crs {vec_crs}!")
    resultGdf = resultGdf.to_crs(vec_crs).rename(columns=renamingPairs)
    if savenameRawExtraction is not None:
        if progress:
            print("Storing extraction result to disk!")
        if savetypeRawExtraction=="GPKG":
            resultGdfStringified = utils.stringifyGdf(resultGdf, object_columns=None, geometry_col='geometry')
            resultGdfStringified.to_file(Path(savenameRawExtraction).with_suffix(".gpkg"), driver="GPKG")
        elif savetypeRawExtraction=="GeoJSON":
            resultGdfStringified = utils.stringifyGdf(resultGdf, object_columns=None, geometry_col='geometry')
            resultGdfStringified.to_file(Path(savenameRawExtraction).with_suffix(".geojson"), driver="GeoJSON")
        else:
            resultGdf.to_file(savenameRawExtraction, driver=savetypeRawExtraction)
    return resultGdf

def extractQuantity(
    rasterInput, # list of file paths (local or /vsis3/ or s3://)
    polygonInput, # local path or GeoDataFrame
    weightInput=None, # None, list of paths, or trueAreaForFirst or trueAreaForEach
    quantities=["median"],
    centroidBasedQuantities=[],
    ignoreValues=[],
    
    keepColumns=None, 
    idColumn=None,
    idColumnFallbackName="id",
    savenameRawExtraction=None,
    savetypeRawExtraction="GPKG",
    savenameAreaCalculation=None,
    blocksizeAreaCalculation=None, # None -> infers blocksize from source file; only change if you know what you are doing
    pointsPerEdgeAreaCalculation=1,
    nWorkersAreaCalculation=0,
    
    keepRawExtraction=False,
    keepNonIntersecting=False,
    keepNonIntersecting_mode="allBandsNotIntersecting",
    
    key=None,
    secret=None,
    s3_endpoint=None, # host-like value, e.g. "s3.eu-central-1.amazonaws.com" or "my-s3.example.com"
    requester_pays=False,
    aws_region=None,
    enforceS3Session=False,
    allowPythonVrt=False,
    
    progress = True,
    **exactextract_kwargs
):
    if len(quantities)==0 and not keepRawExtraction:
        raise RuntimeError("At least one quantity has to be chosen or keepRawExtraction has to be True!")
    
    # determine if centroid based
    ops = ["values", "coverage", "weights"]
    if centroidBasedQuantities:
        ops += ["center_x", "center_y"]
    
    # get values and weights
    resultGdf = extractChosenOps(
        rasterInput = rasterInput,          
        polygonInput = polygonInput,        
        weightInput = weightInput, 
        ops = ops, 
        keepColumns = keepColumns,
        idColumn=idColumn,
        idColumnFallbackName=idColumnFallbackName,
        savenameRawExtraction=savenameRawExtraction,
        savetypeRawExtraction=savetypeRawExtraction,
        savenameAreaCalculation=savenameAreaCalculation,
        blocksizeAreaCalculation=blocksizeAreaCalculation,
        pointsPerEdgeAreaCalculation=pointsPerEdgeAreaCalculation,
        nWorkersAreaCalculation=nWorkersAreaCalculation,
        key=key,
        secret=secret,
        s3_endpoint=s3_endpoint, 
        requester_pays=requester_pays,
        aws_region=aws_region,
        enforceS3Session=enforceS3Session,
        allowPythonVrt=allowPythonVrt,
        progress=progress,
        **exactextract_kwargs
    )
    
    if centroidBasedQuantities:
        if progress:
            print("Computing centroid based coverage!")
        columnNamesPerBand = utils._getColumnNamesPerBand(rasterInput, ops)
        resultGdf = utils.computeCentroidCoverage(gdf=resultGdf, columnNamesPerBand=columnNamesPerBand, desiredSubscript="_centroidCoverage", geometryColumn="geometry", progress=progress)
    
    if progress:
        print("Computing chosen quantities!")
    return processRawExtractionToQuantity(
            rasterInput=rasterInput, 
            resultGdf=resultGdf, 
            keepRawExtraction=keepRawExtraction,
            keepNonIntersecting=keepNonIntersecting,
            keepNonIntersecting_mode=keepNonIntersecting_mode,
            quantities=quantities,
            centroidBasedQuantities=centroidBasedQuantities,
            ignoreValues=ignoreValues,
            progress=progress
    )

def processRawExtractionToQuantity(
        rasterInput, 
        resultGdf, 
        keepRawExtraction=False,
        keepNonIntersecting=False,
        keepNonIntersecting_mode="allBandsNotIntersecting",
        quantities=[],
        centroidBasedQuantities=[],
        ignoreValues=[],
        progress=True
):
    if len(quantities)==0 and not keepRawExtraction:
        raise RuntimeError("At least one quantity has to be chosen or keepRawExtraction has to be True!")
    
    ### coverage based quantities ###
    dictionaryQuantities = [q for q in quantities if isinstance(q, dict)]
    stringQuantities = [q for q in quantities if isinstance(q, str)]
    
    funs = []
    desiredSubscripts = []
    tags = []
    for q in dictionaryQuantities:
        # empirical dist
        if "empDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["empDist"])
            funs.append(partial(statUtils.buildEmpDist, **qKwargs))
            desiredSubscripts.append("_empDist")
            tags.append(None)
                
        if "unweightedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedEmpDist"])
            funs.append(partial(statUtils.buildUnweightedEmpDist, **qKwargs))
            desiredSubscripts.append("_unweightedEmpDist")
            tags.append(None)
        
        if "fullyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["fullyContainedEmpDist"])
            funs.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterFullyContained, **qKwargs))
            desiredSubscripts.append("_fullyContainedEmpDist")
            tags.append(None)
            
        if "unweightedFullyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedFullyContainedEmpDist"])
            funs.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterFullyContained, **qKwargs))
            desiredSubscripts.append("_unweightedFullyContainedEmpDist")
            tags.append(None)
        
        if "partiallyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["partiallyContainedEmpDist"])
            funs.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterPartiallyContained, **qKwargs))
            desiredSubscripts.append("_partiallyContainedEmpDist")
            tags.append(None)
        
        if "unweightedPartiallyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedPartiallyContainedEmpDist"])
            funs.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterPartiallyContained, **qKwargs))
            desiredSubscripts.append("_unweightedPartiallyContainedEmpDist")
            tags.append(None)
            
        if "boundaryEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["boundaryEmpDist"])
            funs.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterBoundary, **qKwargs))
            desiredSubscripts.append("_boundaryEmpDist")
            tags.append(None)
            
        if "unweightedBoundaryEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedBoundaryEmpDist"])
            funs.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterBoundary, **qKwargs))
            desiredSubscripts.append("_unweightedBoundaryEmpDist")
            tags.append(None)
                
        if "allTouchedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["allTouchedEmpDist"])
            funs.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterAllTouched, **qKwargs))
            desiredSubscripts.append("_allTouchedEmpDist")
            tags.append(None)
            
        if "unweightedAllTouchedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedAllTouchedEmpDist"])
            funs.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterAllTouched, **qKwargs))
            desiredSubscripts.append("_unweightedAllTouchedEmpDist")
            tags.append(None)
            
        # empirical digest
        if "empDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["empDigest"])
            funs.append(partial(statUtils.buildEmpDigest, **qKwargs))
            desiredSubscripts.append("_empDigest")
            tags.append(None)
                
        if "unweightedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedEmpDigest"])
            funs.append(partial(statUtils.buildUnweightedEmpDigest, **qKwargs))
            desiredSubscripts.append("_unweightedEmpDigest")
            tags.append(None)
        
        if "fullyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["fullyContainedEmpDigest"])
            funs.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterFullyContained, **qKwargs))
            desiredSubscripts.append("_fullyContainedEmpDigest")
            tags.append(None)
            
        if "unweightedFullyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedFullyContainedEmpDigest"])
            funs.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterFullyContained, **qKwargs))
            desiredSubscripts.append("_unweightedFullyContainedEmpDigest")
            tags.append(None)
        
        if "partiallyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["partiallyContainedEmpDigest"])
            funs.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterPartiallyContained, **qKwargs))
            desiredSubscripts.append("_partiallyContainedEmpDigest")
            tags.append(None)
        
        if "unweightedPartiallyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedPartiallyContainedEmpDigest"])
            funs.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterPartiallyContained, **qKwargs))
            desiredSubscripts.append("_unweightedPartiallyContainedEmpDigest")
            tags.append(None)
            
        if "boundaryEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["boundaryEmpDigest"])
            funs.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterBoundary, **qKwargs))
            desiredSubscripts.append("_boundaryEmpDigest")
            tags.append(None)
            
        if "unweightedBoundaryEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedBoundaryEmpDigest"])
            funs.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterBoundary, **qKwargs))
            desiredSubscripts.append("_unweightedBoundaryEmpDigest")
            tags.append(None)
                
        if "allTouchedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["allTouchedEmpDigest"])
            funs.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterAllTouched, **qKwargs))
            desiredSubscripts.append("_allTouchedEmpDigest")
            tags.append(None)
            
        if "unweightedAllTouchedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedAllTouchedEmpDigest"])
            funs.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterAllTouched, **qKwargs))
            desiredSubscripts.append("_unweightedAllTouchedEmpDigest")
            tags.append(None)
    
    # quantiles (e.g. quantile-0p5 or unweightedQuantile-0p5 or ...)
    quantiles = [utils.parseQuantile(s, base="quantile") for s in stringQuantities]
    for q in quantiles:
        if q is not None:
            funs.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues))
            desiredSubscripts.append("_quantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    unweightedQuantiles = [utils.parseQuantile(s, base="unweightedQuantile") for s in stringQuantities]
    for q in unweightedQuantiles:
        if q is not None:
            funs.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues))
            desiredSubscripts.append("_unweightedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
            
    fullyContainedQuantiles = [utils.parseQuantile(s, base="fullyContainedQuantile") for s in stringQuantities]
    for q in fullyContainedQuantiles:
        if q is not None:
            funs.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
            desiredSubscripts.append("_fullyContainedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    unweightedFullyContainedQuantiles = [utils.parseQuantile(s, base="unweightedFullyContainedQuantile") for s in stringQuantities]
    for q in unweightedFullyContainedQuantiles:
        if q is not None:
            funs.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
            desiredSubscripts.append("_unweightedFullyContainedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
            
    partiallyContainedQuantiles = [utils.parseQuantile(s, base="partiallyContainedQuantile") for s in stringQuantities]
    for q in partiallyContainedQuantiles:
        if q is not None:
            funs.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
            desiredSubscripts.append("_partiallyContainedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    unweightedPartiallyContainedQuantiles = [utils.parseQuantile(s, base="unweightedPartiallyContainedQuantile") for s in stringQuantities]
    for q in unweightedPartiallyContainedQuantiles:
        if q is not None:
            funs.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
            desiredSubscripts.append("_unweightedPartiallyContainedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
            
    boundaryQuantiles = [utils.parseQuantile(s, base="boundaryQuantile") for s in stringQuantities]
    for q in boundaryQuantiles:
        if q is not None:
            funs.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
            desiredSubscripts.append("_boundaryQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    unweightedBoundaryQuantiles = [utils.parseQuantile(s, base="unweightedBoundaryQuantile") for s in stringQuantities]
    for q in unweightedBoundaryQuantiles:
        if q is not None:
            funs.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
            desiredSubscripts.append("_unweightedBoundaryQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
            
    allTouchedQuantiles = [utils.parseQuantile(s, base="allTouchedQuantile") for s in stringQuantities]
    for q in allTouchedQuantiles:
        if q is not None:
            funs.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
            desiredSubscripts.append("_allTouchedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    unweightedAllTouchedQuantiles = [utils.parseQuantile(s, base="unweightedAllTouchedQuantile") for s in stringQuantities]
    for q in unweightedAllTouchedQuantiles:
        if q is not None:
            funs.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
            desiredSubscripts.append("_unweightedAllTouchedQuantile-" + f"{q}".replace(".", "p"))
            tags.append(None)
    
    # median stuff
    if "median" in stringQuantities:
        funs.append(partial(statUtils.median, ignoreValues=ignoreValues))
        desiredSubscripts.append("_median")
        tags.append(None)
    
    if "unweightedMedian" in stringQuantities:
        funs.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues))
        desiredSubscripts.append("_unweightedMedian")
        tags.append(None)
        
    if "fullyContainedMedian" in stringQuantities:
        funs.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_fullyContainedMedian")
        tags.append(None)
    
    if "unweightedFullyContainedMedian" in stringQuantities:
        funs.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_unweightedFullyContainedMedian")
        tags.append(None)
        
    if "partiallyContainedMedian" in stringQuantities:
        funs.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_partiallyContainedMedian")
        tags.append(None)
    
    if "unweightedPartiallyContainedMedian" in stringQuantities:
        funs.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_unweightedPartiallyContainedMedian")
        tags.append(None)
        
    if "boundaryMedian" in stringQuantities:
        funs.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_boundaryMedian")
        tags.append(None)
    
    if "unweightedBoundaryMedian" in stringQuantities:
        funs.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_unweightedBoundaryMedian")
        tags.append(None)
    
    if "allTouchedMedian" in stringQuantities:
        funs.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_allTouchedMedian")
        tags.append(None)
    
    if "unweightedAllTouchedMedian" in stringQuantities:
        funs.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_unweightedAllTouchedMedian")
        tags.append(None)
    
    # mean
    if "mean" in stringQuantities:
        funs.append(partial(statUtils.mean, ignoreValues=ignoreValues))
        desiredSubscripts.append("_mean")
        tags.append(None)
    
    if "unweightedMean" in stringQuantities:
        funs.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues))
        desiredSubscripts.append("_unweightedMean")
        tags.append(None)
    
    if "fullyContainedMean" in stringQuantities:
        funs.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_fullyContainedMean")
        tags.append(None)
    
    if "unweightedFullyContainedMean" in stringQuantities:
        funs.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_unweightedFullyContainedMean")
        tags.append(None)
    
    if "partiallyContainedMean" in stringQuantities:
        funs.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_partiallyContainedMean")
        tags.append(None)
    
    if "unweightedPartiallyContainedMean" in stringQuantities:
        funs.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_unweightedPartiallyContainedMean")
        tags.append(None)
    
    if "boundaryMean" in stringQuantities:
        funs.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_boundaryMean")
        tags.append(None)
    
    if "unweightedBoundaryMean" in stringQuantities:
        funs.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_unweightedBoundaryMean")
        tags.append(None)
        
    if "allTouchedMean" in stringQuantities:
        funs.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_allTouchedMean")
        tags.append(None)
    
    if "unweightedAllTouchedMean" in stringQuantities:
        funs.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_unweightedAllTouchedMean")
        tags.append(None)
        
    # variance
    if "var" in stringQuantities:
        funs.append(partial(statUtils.var, ignoreValues=ignoreValues))
        desiredSubscripts.append("_var")
        tags.append(None)

    if "unweightedVar" in stringQuantities:
        funs.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues))
        desiredSubscripts.append("_unweightedVar")
        tags.append(None)
        
    if "fullyContainedVar" in stringQuantities:
        funs.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_fullyContainedVar")
        tags.append(None)
    
    if "unweightedFullyContainedVar" in stringQuantities:
        funs.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_unweightedFullyContainedVar")
        tags.append(None)
        
    if "partiallyContainedVar" in stringQuantities:
        funs.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_partiallyContainedVar")
        tags.append(None)
    
    if "unweightedPartiallyContainedVar" in stringQuantities:
        funs.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_unweightedPartiallyContainedVar")
        tags.append(None)
        
    if "boundaryVar" in stringQuantities:
        funs.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_boundaryVar")
        tags.append(None)
    
    if "unweightedBoundaryVar" in stringQuantities:
        funs.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_unweightedBoundaryVar")
        tags.append(None)
    
    if "allTouchedVar" in stringQuantities:
        funs.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_allTouchedVar")
        tags.append(None)
    
    if "unweightedAllTouchedVar" in stringQuantities:
        funs.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_unweightedAllTouchedVar")
        tags.append(None)
        
    # std
    if "std" in stringQuantities:
        funs.append(partial(statUtils.std, ignoreValues=ignoreValues))
        desiredSubscripts.append("_std")
        tags.append(None)

    if "unweightedStd" in stringQuantities:
        funs.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues))
        desiredSubscripts.append("_unweightedStd")
        tags.append(None)
    
    if "fullyContainedStd" in stringQuantities:
        funs.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_fullyContainedStd")
        tags.append(None)
    
    if "unweightedFullyContainedStd" in stringQuantities:
        funs.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_unweightedFullyContainedStd")
        tags.append(None)
        
    if "partiallyContainedStd" in stringQuantities:
        funs.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_partiallyContainedStd")
        tags.append(None)
    
    if "unweightedPartiallyContainedStd" in stringQuantities:
        funs.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_unweightedPartiallyContainedStd")
        tags.append(None)
        
    if "boundaryStd" in stringQuantities:
        funs.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_boundaryStd")
        tags.append(None)
    
    if "unweightedBoundaryStd" in stringQuantities:
        funs.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_unweightedBoundaryStd")
        tags.append(None)
    
    if "allTouchedStd" in stringQuantities:
        funs.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_allTouchedStd")
        tags.append(None)
    
    if "unweightedAllTouchedStd" in stringQuantities:
        funs.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_unweightedAllTouchedStd")   
        tags.append(None)
    
    # weight
    if "weight" in stringQuantities:
        funs.append(partial(statUtils.weight, ignoreValues=ignoreValues))
        desiredSubscripts.append("_weight")
        tags.append(None)

    if "unweightedWeight" in stringQuantities:
        funs.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues))
        desiredSubscripts.append("_unweightedWeight")
        tags.append(None)
    
    if "fullyContainedWeight" in stringQuantities:
        funs.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_fullyContainedWeight")
        tags.append(None)
    
    if "unweightedFullyContainedWeight" in stringQuantities:
        funs.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        desiredSubscripts.append("_unweightedFullyContainedWeight")
        tags.append(None)
        
    if "partiallyContainedWeight" in stringQuantities:
        funs.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_partiallyContainedWeight")
        tags.append(None)
    
    if "unweightedPartiallyContainedWeight" in stringQuantities:
        funs.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterPartiallyContained))
        desiredSubscripts.append("_unweightedPartiallyContainedWeight")
        tags.append(None)
        
    if "boundaryWeight" in stringQuantities:
        funs.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_boundaryWeight")
        tags.append(None)
    
    if "unweightedBoundaryWeight" in stringQuantities:
        funs.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        desiredSubscripts.append("_unweightedBoundaryWeight")
        tags.append(None)
    
    if "allTouchedWeight" in stringQuantities:
        funs.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_allTouchedWeight")
        tags.append(None)
    
    if "unweightedAllTouchedWeight" in stringQuantities:
        funs.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterAllTouched))
        desiredSubscripts.append("_unweightedAllTouchedWeight")   
        tags.append(None)
        
    ### centroid based quantities ###
    centroidBasedDictionaryQuantities = [q for q in centroidBasedQuantities if isinstance(q, dict)]
    centroidBasedStringQuantities = [q for q in centroidBasedQuantities if isinstance(q, str)]
    
    centroidBasedFuns = []
    centroidBasedDesiredSubscripts = []
    centroidBasedTags = []
    for q in centroidBasedDictionaryQuantities:
        # empirical dist
        if "centroidBasedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDist, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedEmpDist")
            centroidBasedTags.append("centroidBased")
                
        if "unweightedCentroidBasedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDist, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedEmpDist")
            centroidBasedTags.append("centroidBased")
            
        if "centroidBasedFullyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedFullyContainedEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterFullyContained, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedEmpDist")
            centroidBasedTags.append("partialCentroidBased")
            
        if "unweightedCentroidBasedFullyContainedEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedFullyContainedEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterFullyContained, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedEmpDist")
            centroidBasedTags.append("partialCentroidBased")
        
        if "centroidBasedBoundaryEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedBoundaryEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDist, filterFun=statUtils.filterBoundary, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryEmpDist")
            centroidBasedTags.append("partialCentroidBased")
            
        if "unweightedCentroidBasedBoundaryEmpDist" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedBoundaryEmpDist"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDist, filterFun=statUtils.filterBoundary, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryEmpDist")
            centroidBasedTags.append("partialCentroidBased")        
            
        # empirical digest
        if "centroidBasedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDigest, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedEmpDigest")
            centroidBasedTags.append("centroidBased")
                
        if "unweightedCentroidBasedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDigest, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedEmpDigest")
            centroidBasedTags.append("centroidBased")
            
        if "centroidBasedFullyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedFullyContainedEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterFullyContained, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedEmpDigest")
            centroidBasedTags.append("partialCentroidBased")
            
        if "unweightedCentroidBasedFullyContainedEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedFullyContainedEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterFullyContained, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedEmpDigest")
            centroidBasedTags.append("partialCentroidBased")
        
        if "centroidBasedBoundaryEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["centroidBasedBoundaryEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildEmpDigest, filterFun=statUtils.filterBoundary, **qKwargs))
            centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryEmpDigest")
            centroidBasedTags.append("partialCentroidBased")
            
        if "unweightedCentroidBasedBoundaryEmpDigest" in q.keys():
            qKwargs = {"ignoreValues":ignoreValues}
            qKwargs.update(**q["unweightedCentroidBasedBoundaryEmpDigest"])
            centroidBasedFuns.append(partial(statUtils.buildUnweightedEmpDigest, filterFun=statUtils.filterBoundary, **qKwargs))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryEmpDigest")
            centroidBasedTags.append("partialCentroidBased")   
    
    # quantiles (e.g. quantile-0p5 or unweightedQuantile-0p5 or allTouchedQuantile-0p5)    
    quantiles = [utils.parseQuantile(s, base="centroidBasedQuantile") for s in centroidBasedStringQuantities]
    for q in quantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues))
            centroidBasedDesiredSubscripts.append("_centroidBasedQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("centroidBased")
    
    unweightedQuantiles = [utils.parseQuantile(s, base="unweightedCentroidBasedQuantile") for s in centroidBasedStringQuantities]
    for q in unweightedQuantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("centroidBased")
            
    quantiles = [utils.parseQuantile(s, base="centroidBasedFullyContainedQuantile") for s in centroidBasedStringQuantities]
    for q in quantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
            centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("partialCentroidBased")
            
    unweightedQuantiles = [utils.parseQuantile(s, base="unweightedCentroidBasedFullyContainedQuantile") for s in centroidBasedStringQuantities]
    for q in unweightedQuantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("partialCentroidBased")
    
    quantiles = [utils.parseQuantile(s, base="centroidBasedBoundaryQuantile") for s in centroidBasedStringQuantities]
    for q in quantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_quantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
            centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("partialCentroidBased")
            
    unweightedQuantiles = [utils.parseQuantile(s, base="unweightedCentroidBasedBoundaryQuantile") for s in centroidBasedStringQuantities]
    for q in unweightedQuantiles:
        if q is not None:
            centroidBasedFuns.append(statUtils.make_unweightedQuantile_fn(q, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
            centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryQuantile-" + f"{q}".replace(".", "p"))
            centroidBasedTags.append("partialCentroidBased")

    # median stuff
    if "centroidBasedMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.median, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_centroidBasedMedian")
        centroidBasedTags.append("centroidBased")
    
    if "unweightedCentroidBasedMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedMedian")
        centroidBasedTags.append("centroidBased")
        
    if "centroidBasedFullyContainedMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedMedian")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedFullyContainedMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedMedian")
        centroidBasedTags.append("partialCentroidBased")
        
    if "centroidBasedBoundaryMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.median, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryMedian")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedBoundaryMedian" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMedian, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryMedian")
        centroidBasedTags.append("partialCentroidBased")

    # mean
    if "centroidBasedMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.mean, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_centroidBasedMean")
        centroidBasedTags.append("centroidBased")
    
    if "unweightedCentroidBasedMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedMean")
        centroidBasedTags.append("centroidBased")
        
    if "centroidBasedFullyContainedMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedMean")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedFullyContainedMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedMean")
        centroidBasedTags.append("partialCentroidBased")
        
    if "centroidBasedBoundaryMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.mean, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryMean")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedBoundaryMean" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedMean, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryMean")
        centroidBasedTags.append("partialCentroidBased")
        
    # variance
    if "centroidBasedVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.var, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_centroidBasedVar")
        centroidBasedTags.append("centroidBased")

    if "unweightedCentroidBasedVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedVar")
        centroidBasedTags.append("centroidBased")
    
    if "centroidBasedFullyContainedVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedVar")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedFullyContainedVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedVar")
        centroidBasedTags.append("partialCentroidBased")
    
    if "centroidBasedBoundaryVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.var, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryVar")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedBoundaryVar" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedVar, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryVar")
        centroidBasedTags.append("partialCentroidBased")
        
    # std
    if "centroidBasedStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.std, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_centroidBasedStd")
        centroidBasedTags.append("centroidBased")

    if "unweightedCentroidBasedStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedStd")
        centroidBasedTags.append("centroidBased")
    
    if "centroidBasedFullyContainedStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedStd")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedFullyContainedStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedStd")
        centroidBasedTags.append("partialCentroidBased")
    
    if "centroidBasedBoundaryStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.std, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryStd")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedBoundaryStd" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedStd, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryStd")
        centroidBasedTags.append("partialCentroidBased")
    
    # weight
    if "centroidBasedWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.weight, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_centroidBasedWeight")
        centroidBasedTags.append("centroidBased")

    if "unweightedCentroidBasedWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedWeight")
        centroidBasedTags.append("centroidBased")
    
    if "centroidBasedFullyContainedWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_centroidBasedFullyContainedWeight")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedFullyContainedWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterFullyContained))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedFullyContainedWeight")
        centroidBasedTags.append("partialCentroidBased")
    
    if "centroidBasedBoundaryWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.weight, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_centroidBasedBoundaryWeight")
        centroidBasedTags.append("partialCentroidBased")
    
    if "unweightedCentroidBasedBoundaryWeight" in centroidBasedStringQuantities:
        centroidBasedFuns.append(partial(statUtils.unweightedWeight, ignoreValues=ignoreValues, filterFun=statUtils.filterBoundary))
        centroidBasedDesiredSubscripts.append("_unweightedCentroidBasedBoundaryWeight")
        centroidBasedTags.append("partialCentroidBased")
        
    return processRawExtraction(
            rasterInput=rasterInput, 
            resultGdf=resultGdf, 
            keepRawExtraction=keepRawExtraction,
            keepNonIntersecting=keepNonIntersecting,
            keepNonIntersecting_mode=keepNonIntersecting_mode,
            funs=funs,
            desiredSubscripts=desiredSubscripts,
            tags=tags,
            centroidBasedFuns=centroidBasedFuns,
            centroidBasedDesiredSubscripts=centroidBasedDesiredSubscripts,
            centroidBasedTags=centroidBasedTags,
            progress=progress
    )

def processRawExtraction(
        rasterInput, 
        resultGdf, 
        keepRawExtraction=False,
        keepNonIntersecting=False,
        keepNonIntersecting_mode="allBandsNotIntersecting",
        funs=[],
        desiredSubscripts=[],
        tags=[],
        centroidBasedFuns=[],
        centroidBasedDesiredSubscripts=[],
        centroidBasedTags=[],
        progress=True
):
    if (len(funs)==0 or len(desiredSubscripts)==0) and (len(centroidBasedFuns)==0 or len(centroidBasedDesiredSubscripts)==0):
        raise RuntimeError("At least one of function/subscript pair has to be given!")
    
    ops = ["values", "coverage", "weights"]
    if len(centroidBasedFuns)>0:
        ops += ["centroidCoverage", "center_x", "center_y"]
    columnNamesPerBand = utils._getColumnNamesPerBand(rasterInput, ops)
    
    if keepNonIntersecting is False:
        toDropIdx = []
        for r, row in resultGdf.iterrows():
            if keepNonIntersecting_mode=="allBandsNotIntersecting": # checks if there are non-empty values for any of the rasterInput
                decider = np.sum([len(row[columns["values"]]) for bandName, columns in columnNamesPerBand.items()])==0
            else: # checks if there are any empty values for any of the rasterInput
                decider = np.prod([len(row[columns["values"]]) for bandName, columns in columnNamesPerBand.items()])==0
            if decider: 
                toDropIdx.append(r)
        resultGdf = resultGdf.drop(index=toDropIdx)
    
    for b, (bandName, columns) in enumerate(columnNamesPerBand.items()):
        if progress:
            print(f"Start processing of band {b+1} of in total {len(columnNamesPerBand)} bands!")
        for fun, desiredSubscript, tag in zip(funs, desiredSubscripts, tags):
            _processColumn(resultGdf, bandName, columns, desiredSubscript=desiredSubscript, fun=fun, progress=progress)
        for fun, desiredSubscript, tag in zip(centroidBasedFuns, centroidBasedDesiredSubscripts, centroidBasedTags):
            if tag=="partialCentroidBased":
                _processColumnPartialCentroidBased(resultGdf, bandName, columns, desiredSubscript=desiredSubscript, fun=fun, progress=progress)
            else:
                _processColumnCentroidBased(resultGdf, bandName, columns, desiredSubscript=desiredSubscript, fun=fun, progress=progress)
            
    if keepRawExtraction is False:
        rawColumns = []
        for columns in columnNamesPerBand.values():
            rawColumns.extend(columns.values())
        resultGdf.drop(columns=rawColumns, inplace=True)
    
    return resultGdf

def _processColumn(resultGdf, bandName, columns, desiredSubscript, fun, desc=None, progress=False):
    if progress:
        print("Processing", bandName+desiredSubscript, "using the", fun.func.__name__ if isinstance(fun, partial) else fun.__name__, "function:")
        resultGdf[bandName+desiredSubscript] = resultGdf.progress_apply(lambda row: fun(row[columns["values"]], row[columns["coverage"]], row[columns["weights"]]), axis=1)
    else:
        resultGdf[bandName+desiredSubscript] = resultGdf.apply(lambda row: fun(row[columns["values"]], row[columns["coverage"]], row[columns["weights"]]), axis=1)
        
def _processColumnCentroidBased(resultGdf, bandName, columns, desiredSubscript, fun, desc=None, progress=False):
    if progress:
        print("Processing", bandName+desiredSubscript, "using the", fun.func.__name__ if isinstance(fun, partial) else fun.__name__, "function:")
        resultGdf[bandName+desiredSubscript] = resultGdf.progress_apply(lambda row: fun(row[columns["values"]], row[columns["centroidCoverage"]], row[columns["weights"]]), axis=1)
    else:
        resultGdf[bandName+desiredSubscript] = resultGdf.apply(lambda row: fun(row[columns["values"]], row[columns["centroidCoverage"]], row[columns["weights"]]), axis=1)

def _processColumnPartialCentroidBased(resultGdf, bandName, columns, desiredSubscript, fun, desc=None, progress=False):
    if progress:
        print("Processing", bandName+desiredSubscript, "using the", fun.func.__name__ if isinstance(fun, partial) else fun.__name__, "function:")
        resultGdf[bandName+desiredSubscript] = resultGdf.progress_apply(
            lambda row: fun(row[columns["values"]], np.array(row[columns["centroidCoverage"]])*np.array(row[columns["coverage"]]), row[columns["weights"]]), axis=1
        )
    else:
        resultGdf[bandName+desiredSubscript] = resultGdf.apply(
            lambda row: fun(row[columns["values"]], np.array(row[columns["centroidCoverage"]])*np.array(row[columns["coverage"]]), row[columns["weights"]]), axis=1
        )
        