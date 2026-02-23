import numpy as np
import uuid
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.env import Env
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyproj import Transformer, CRS
from typing import Tuple
from tqdm import tqdm

from .libs import utils

def _densified_vertices_for_block(xs: np.ndarray, ys: np.ndarray, points_per_edge: int) -> np.ndarray:
    """
    Build densified vertex arrays for every pixel in block.
    xs, ys shape: (nrows+1, ncols+1)
    Returns: verts shape (num_pixels, nverts, 2) with vertex order [top edge -> right -> bottom -> left]
    """
    nrows = xs.shape[0] - 1
    ncols = xs.shape[1] - 1
    ULx = xs[:-1, :-1]  # (nrows, ncols)
    URx = xs[:-1, 1:]
    LLx = xs[1:, :-1]
    LRx = xs[1:, 1:]
    ULy = ys[:-1, :-1]
    URy = ys[:-1, 1:]
    LLy = ys[1:, :-1]
    LRy = ys[1:, 1:]

    k = int(points_per_edge) + 1  # points per edge including both corners
    # t values including both endpoints but we'll avoid duplicating corners when concatenating edges
    t = np.linspace(0.0, 1.0, k, dtype=float)  # shape (k,)

    # For each edge produce shape (nrows, ncols, k)
    # top edge UL -> UR
    top_x = ULx[:, :, None] * (1.0 - t[None, None, :]) + URx[:, :, None] * t[None, None, :]
    top_y = ULy[:, :, None] * (1.0 - t[None, None, :]) + URy[:, :, None] * t[None, None, :]

    # right edge UR -> LR
    right_x = URx[:, :, None] * (1.0 - t[None, None, :]) + LRx[:, :, None] * t[None, None, :]
    right_y = URy[:, :, None] * (1.0 - t[None, None, :]) + LRy[:, :, None] * t[None, None, :]

    # bottom edge LR -> LL (note direction clockwise)
    bottom_x = LRx[:, :, None] * (1.0 - t[None, None, :]) + LLx[:, :, None] * t[None, None, :]
    bottom_y = LRy[:, :, None] * (1.0 - t[None, None, :]) + LLy[:, :, None] * t[None, None, :]

    # left edge LL -> UL
    left_x = LLx[:, :, None] * (1.0 - t[None, None, :]) + ULx[:, :, None] * t[None, None, :]
    left_y = LLy[:, :, None] * (1.0 - t[None, None, :]) + ULy[:, :, None] * t[None, None, :]

    # Concatenate edges while avoiding repeated corner points:
    # keep all k points from top, keep k-1 points from right (excluding first corner),
    # keep k-1 from bottom, keep k-1 from left -> total verts = k + 3*(k-1) = 4*k - 3
    def trim_edge(arr):
        return arr[:, :, 1:]  # drop first point to avoid duplicating corner when concatenating

    verts_x = np.concatenate([top_x, trim_edge(right_x), trim_edge(bottom_x), trim_edge(left_x)], axis=2)
    verts_y = np.concatenate([top_y, trim_edge(right_y), trim_edge(bottom_y), trim_edge(left_y)], axis=2)

    nverts = verts_x.shape[2]
    # reshape to (num_pixels, nverts, 2)
    num_pixels = nrows * ncols
    verts = np.empty((num_pixels, nverts, 2), dtype=float)
    verts[:, :, 0] = verts_x.reshape(num_pixels, nverts)
    verts[:, :, 1] = verts_y.reshape(num_pixels, nverts)
    return verts  # (num_pixels, nverts, 2)

def _vectorized_shoelace_area(coords: np.ndarray) -> np.ndarray:
    """
    coords: (num_polygons, nverts, 2)
    returns: areas shape (num_polygons,)
    """
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    # roll to get x_{i+1}, y_{i+1}
    xr = np.roll(x, -1, axis=1)
    yr = np.roll(y, -1, axis=1)
    cross = x * yr - xr * y
    area = 0.5 * np.abs(np.sum(cross, axis=1))
    return area  # (num_polygons,)

def _compute_block_vectorized(src_path: str,
                              row_off: int, col_off: int,
                              nrows: int, ncols: int,
                              points_per_edge: int,
                              transform,
                              src_crs) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Worker function: compute array of per-pixel areas for the block using vectorized operations.
    Returns row_off, col_off, nrows, ncols, areas_flat
    """
    # build corner coordinate arrays via affine (robust, fast)
    row_idxs = np.arange(row_off, row_off + nrows + 1)
    col_idxs = np.arange(col_off, col_off + ncols + 1)
    a, b, c = transform.a, transform.b, transform.c
    d, e, f = transform.d, transform.e, transform.f
    xs = (c + col_idxs[None, :] * a + row_idxs[:, None] * b).astype(float)
    ys = (f + col_idxs[None, :] * d + row_idxs[:, None] * e).astype(float)

    # densify per pixel, returns shape (num_pixels, nverts, 2)
    verts_src = _densified_vertices_for_block(xs, ys, points_per_edge)  # in source CRS

    # Build transformer from source CRS to local LAEA
    # compute block center coordinates in source CRS
    center_row = row_off + nrows / 2.0
    center_col = col_off + ncols / 2.0
    cx = float(c + center_col * a + center_row * b)
    cy = float(f + center_col * d + center_row * e)
    if src_crs and getattr(src_crs, "is_geographic", False):
        lon0, lat0 = cx, cy
    else:
        to_geo = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon0, lat0 = to_geo.transform(cx, cy)
    laea_proj4 = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    laea = CRS.from_proj4(laea_proj4)
    transformer = Transformer.from_crs(src_crs, laea, always_xy=True)

    # batch transform: flatten vertex arrays
    num_polys, nverts, _ = verts_src.shape
    xs_flat = verts_src[:, :, 0].ravel()
    ys_flat = verts_src[:, :, 1].ravel()
    # pyproj accepts 1D arrays and returns arrays of same length
    xs_t, ys_t = transformer.transform(xs_flat, ys_flat)
    xs_t = np.asarray(xs_t).reshape(num_polys, nverts)
    ys_t = np.asarray(ys_t).reshape(num_polys, nverts)
    verts_laea = np.stack([xs_t, ys_t], axis=2)  # (num_polys, nverts, 2)

    # compute areas vectorized
    areas = _vectorized_shoelace_area(verts_laea)  # shape (num_polys,)

    # shape back to block (nrows, ncols)
    areas_block = areas.reshape((nrows, ncols)).astype(np.float32)
    return (row_off, col_off, nrows, ncols, areas_block)

def calculatePixelAreaTif(
    src_path: str,
    dst_path: str,
    points_per_edge: int = 1,
    block_size: int | None = None,
    num_workers: int = 4,
    nodata: float = np.nan,
    fallbackBlocksize: int = 512,
    progress: bool = True,
    
    key: str | None = None,
    secret: str | None = None,
    s3_endpoint: str | None = None,
    requester_pays: bool = False,
    aws_region: str | None = None,
) -> str:
    assert points_per_edge>0
    
    if dst_path is None:
        p = Path(src_path)
        dst_path = f"{p.stem}_pixelArea_{uuid.uuid4()}.tif"
    
    # prepare meta and tasks
    if utils._is_s3_path(src_path):
        aws_session, gdal_env = utils.getRasterioAwsSession(key, secret, s3_endpoint, requester_pays, aws_region)
        with Env(session=aws_session, AWS_VIRTUAL_HOSTING=False):
            with utils._temporary_environ(gdal_env):
                with rasterio.open(src_path) as src:
                    profile = src.profile.copy()
                    width, height = src.width, src.height
                    transform = src.transform
                    src_crs = src.crs
            
                    if block_size is None:
                        try:
                            bh, bw = src.block_shapes[0]
                            bh, bw = int(bh), int(bw)
                            if bh%16 or bw%16:
                                raise RuntimeError(f"The internal tiling has to be a multiple of 16 and currently it is width={bw} and height={bh}; hence, we will change to a fallback value of {fallbackBlocksize} and continue!")
                        except Exception as e:
                            print(e)
                            bh = bw = fallbackBlocksize
                    else:
                        bh = bw = int(block_size)
                
    else:
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            width, height = src.width, src.height
            transform = src.transform
            src_crs = src.crs
    
            if block_size is None:
                try:
                    bh, bw = src.block_shapes[0]
                    bh, bw = int(bh), int(bw)
                    if bh%16 or bw%16:
                        raise RuntimeError(f"The internal tiling has to be a multiple of 16 and currently it is width={bw} and height={bh}; hence, we will change to a fallback value of {fallbackBlocksize} and continue!")
                except Exception as e:
                    print(e)
                    bh = bw = fallbackBlocksize
            else:
                bh = bw = int(block_size)
    
    profile.update(driver="GTiff", dtype='float32', count=1, compress='deflate',
                   tiled=True, blockxsize=bw, blockysize=bh, nodata=nodata)

    tasks = []
    for row_off in range(0, height, bh):
        for col_off in range(0, width, bw):
            nrows = min(bh, height - row_off)
            ncols = min(bw, width - col_off)
            tasks.append((src_path, row_off, col_off, nrows, ncols, int(points_per_edge), transform, src_crs))

    # compute in parallel and write results in main process
    with rasterio.open(dst_path, 'w', **profile) as dst:
        if num_workers:
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                futures = [exe.submit(_compute_block_vectorized, *task) for task in tasks]
                for fut in tqdm(as_completed(futures), desc="Blockwise Area Computation", total=len(tasks), disable=not progress):
                    row_off, col_off, nrows, ncols, block = fut.result()
                    dst.write(block, 1, window=Window(col_off, row_off, ncols, nrows))
        else:
            for task in tqdm(tasks, desc="Blockwise Area Computation", disable=not progress):
                row_off, col_off, nrows, ncols, block = _compute_block_vectorized(*task)
                dst.write(block, 1, window=Window(col_off, row_off, ncols, nrows))
    dst.close()

    return dst_path