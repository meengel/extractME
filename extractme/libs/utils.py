import os
from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
import boto3
import re

from shapely.geometry import Point
from shapely.prepared import prep
import geopandas as gpd
import rasterio
from rasterio.session import AWSSession
from tqdm import tqdm
tqdm.pandas()

import importlib
ee = importlib.import_module("exactextract.exact_extract")

def _safe_serialize(obj):
    """Convert NumPy arrays to lists before JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def stringifyGdf(gdf, object_columns=None, geometry_col='geometry'):
    """
    Save a GeoDataFrame with object-type columns to a GeoPackage.
    Object columns are serialized as JSON strings, including NumPy arrays.

    Parameters:
    - gdf: GeoDataFrame to save
    - object_columns: List of column names to serialize (if None, auto-detect object dtype)
    - geometry_col: Name of geometry column (default: 'geometry')
    """
    gdf_copy = gdf.copy()

    # Auto-detect object columns if not provided
    if object_columns is None:
        object_columns = [col for col in gdf_copy.columns
                          if gdf_copy[col].dtype == 'object' and col != geometry_col]

    # Serialize object columns to JSON strings
    for col in object_columns:
        gdf_copy[col] = gdf_copy[col].apply(lambda x: json.dumps(_safe_serialize(x)))

    # Save to GeoPackage
    return gdf_copy

def destringifyGdf(gdf, object_columns=None, geometry_col='geometry'):
    """
    Load a GeoDataFrame from a GeoPackage and deserialize JSON-encoded object columns.

    Parameters:
    - object_columns: List of column names to deserialize (if None, auto-detect JSON-like strings)
    - geometry_col: Name of geometry column (default: 'geometry')

    Returns:
    - GeoDataFrame with deserialized object columns
    """

    # Auto-detect JSON-like columns if not provided
    if object_columns is None:
        object_columns = [col for col in gdf.columns
                          if col != geometry_col and gdf[col].apply(lambda x: isinstance(x, str) and x.startswith('[')).any()]

    # Deserialize JSON strings
    for col in object_columns:
        gdf[col] = gdf[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return gdf

def _lenList(val, desiredLen=1):
    if isinstance(val, list) and desiredLen==len(val):
        return val
    elif isinstance(val, list) and 1==len(val):
        return val*desiredLen
    else:
        return [val]*desiredLen

def _is_s3_path(p):
    if not p:
        return False
    p = str(p)
    return p.startswith("/vsis3/") or p.startswith("s3://") or "/vsis3/" in p

def parseQuantile(s, base="quantile"):
    # PAT = re.compile(r"^"+base+"-([01])p(\d+)$")
    # PAT = re.compile(r"^" + re.escape(base) + r"-([01])p(\d+)$")
    PAT = re.compile(rf"^{re.escape(base)}-([01])p(\d+)$")

    m = PAT.match(s)
    if not m:
        return None
    int_part, frac = m.group(1), m.group(2)
    return float(f"{int_part}.{frac}")

def getRasterioAwsSession(key, secret, s3_endpoint, requester_pays, aws_region):
    # S3 handling: build a boto3 session if credentials are provided (falls back to env/instance profile)
    boto_sess = None
    if key is not None or secret is not None:
        boto_sess = boto3.Session(
            aws_access_key_id = key,
            aws_secret_access_key = secret,
            region_name = aws_region
        )
    else:
        boto_sess = boto3.Session(region_name=aws_region)

    # Create rasterio AWSSession with optional endpoint_url and requester_pays
    aws_session = AWSSession(
        session = boto_sess,
        requester_pays = requester_pays,
        endpoint_url = s3_endpoint,
        region_name = aws_region
    )

    # Prepare GDAL/VSI environment options for S3-compatible endpoint
    # Note: GDAL expects the authority (no scheme) for AWS_S3_ENDPOINT in many setups
    gdal_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "YES",
        "AWS_REQUEST_PAYER": "requester" if requester_pays else None,
    }
    if s3_endpoint:
        # ensure we set the GDAL-side endpoint (authority only, no scheme)
        # common var name is AWS_S3_ENDPOINT per GDAL notes and community guidance
        gdal_env["AWS_S3_ENDPOINT"] = s3_endpoint

    # Optionally expose creds also as env vars for GDAL if explicit keys were passed
    if key is not None:
        gdal_env["AWS_ACCESS_KEY_ID"] = key
    if secret is not None:
        gdal_env["AWS_SECRET_ACCESS_KEY"] = secret
    if aws_region:
        gdal_env["AWS_REGION"] = aws_region
        
    return aws_session, gdal_env

@contextmanager
def _temporary_environ(vars_dict):
    old = {k: os.environ.get(k) for k in vars_dict}
    os.environ.update({k: str(v) for k, v in vars_dict.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _prep_raster_patched(rast, name_root=None):
    # patched version of the original form exactextract such that names are set for a list of raster inputs
    # -> necessary since only by providing a list of rasterio-datasets as an input, it is ensured that the datasets are closed after usage from exactextract!
    if rast is None:
        return [None]

    if isinstance(rast, ee.RasterSource):
        return [rast]

    if type(rast) in (list, tuple):
        if all(isinstance(src, (str, os.PathLike)) for src in rast):
            sources = [
                _prep_raster_patched(src, name_root=os.path.splitext(os.path.basename(src))[0])
                for src in rast
            ]
        else:
            sources = [
                _prep_raster_patched(src) if src is None else
                _prep_raster_patched(src, Path(src.name).stem)
                for src in rast
            ] # patched by Michael Engel
        return list(ee.chain.from_iterable(sources))

    for loader in (ee.prep_raster_gdal, ee.prep_raster_rasterio, ee.prep_raster_xarray):
        sources = loader(rast, name_root)
        if sources:
            return sources

    raise Exception("Unhandled raster datatype")

def computeCentroidCoverage(gdf, columnNamesPerBand, desiredSubscript="_centroidCoverage", geometryColumn="geometry", progress=True):
    gdf = gdf.copy()

    # Prepare geometries once for faster contains checks
    prepared = {i: prep(geom) for i, geom in tqdm(gdf[geometryColumn].items(), desc="Preparing geometries", total=len(gdf), disable=not progress)}

    for b, (bandName, columns) in enumerate(columnNamesPerBand.items()):
        if b==0:
            cx_col = columns["center_x"]
            cy_col = columns["center_y"]
            cov_col = columns["coverage"]  # must exist and align 1:1 with center_x/center_y lists
    
            # Step 1: explode triplets (x, y, coverage) into long form
            if progress:
                print("Forming triplets...")
                triplets = gdf.progress_apply(
                    lambda r: list(zip(r[cx_col], r[cy_col], r[cov_col])),
                    axis=1
                )
            else:
                triplets = gdf.apply(
                    lambda r: list(zip(r[cx_col], r[cy_col], r[cov_col])),
                    axis=1
                )
            
            print("Exploding dataframe...")
            exploded = (
                gdf.reset_index()
                   .assign(triplets=triplets)
                   .explode("triplets", ignore_index=True)
            )
            exploded["center_x"] = exploded["triplets"].str[0]
            exploded["center_y"] = exploded["triplets"].str[1]
            exploded["coverage"] = exploded["triplets"].str[2]
            exploded = exploded.drop(columns="triplets")

            # Step 2: shortcut using per-point coverage
            # coverage == 1.0 -> inside True; otherwise check geometry
            exploded["inside"] = exploded["coverage"]==1.0
            mask = ~exploded["inside"]
            if mask.any():
                # Use prepared parent polygon for the remaining checks
                if progress:
                    print("Checking points...")
                    exploded.loc[mask, "inside"] = exploded.loc[mask].progress_apply(
                        lambda r: prepared[r["index"]].contains(Point(r["center_x"], r["center_y"])), axis=1
                    )
                else:
                    exploded.loc[mask, "inside"] = exploded.loc[mask].apply(
                        lambda r: prepared[r["index"]].contains(Point(r["center_x"], r["center_y"])), axis=1
                    )
    
            # Step 3: regroup results back into lists aligned with original rows
            inside_lists = exploded.groupby("index")["inside"].apply(list)

        # Step 4: assign back to original gdf for this band
        if progress:
            print(f"Assigning centroid coverage to band {bandName}!")
        gdf[bandName + desiredSubscript] = gdf.index.map(inside_lists)

    return gdf

def delete(file, bequiet=False):
    if type(file)==list:
        success = []
        for i in range(len(file)):
            success.append(delete(file[i]))
        return success
    else:
        try:
            os.remove(file)
            return True
        except Exception as e:
            if not bequiet:
                print(e)
                print(f"Removing of {file} did not work! Either it is not existing or you don't have permission for that, e.g. if it is still open in another application!")
            return False

def _getRenamingPairs(rasterInput, operations=["values", "coverage", "weights"]):
    renamingPairs = {}
    for path in rasterInput:
        p = Path(path)
        with rasterio.open(path) as src:
            nBands = src.count
            for b in range(1, nBands+1):
                for op in operations:
                    opName = op.__name__ if callable(op) else op
                    desiredName = f"{p.stem}_band_{b}_{opName}"
                    if nBands==1:
                        if len(rasterInput)==1:
                            currentName = f"{opName}"
                        else:
                            currentName = f"{p.stem}_{opName}"
                    else:
                        if len(rasterInput)==1:
                            currentName = f"band_{b}_{opName}"
                        else:
                            currentName = f"{p.stem}_band_{b}_{opName}"
                    renamingPairs.update({currentName:desiredName})
    return renamingPairs

def _getColumnNamesPerBand(rasterInput, operations=["values", "coverage", "weights"]):
    columnNames = {}
    for path in rasterInput:
        p = Path(path)
        with rasterio.open(path) as src:
            nBands = src.count
            for b in range(1, nBands+1):
                names = {}
                for op in operations:
                    opName = op.__name__ if callable(op) else op
                    names.update({opName: f"{p.stem}_band_{b}_{opName}"})
                columnNames.update({f"{p.stem}_band_{b}": names})
    return columnNames

def getColumnNamesPerBand(rasterInput, rasterInputBands=None, operations=["values", "coverage", "weights"]):
    if rasterInputBands is None:
        rasterInputBands = [1]*len(rasterInput)
    assert len(rasterInputBands)==len(rasterInput)
        
    columnNames = {}
    for path, nBands in zip(rasterInput, rasterInputBands):
        p = Path(path)
        for b in range(1, nBands+1):
            names = {}
            for op in operations:
                opName = op.__name__ if callable(op) else op
                names.update({opName: f"{p.stem}_band_{b}_{opName}"})
            columnNames.update({f"{p.stem}_band_{b}": names})
    return columnNames
