from contextlib import contextmanager, ExitStack
import rasterio
import subprocess

@contextmanager
def openRasterList(pathList, **kwargs):
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(p, **kwargs)) for p in pathList]
        yield datasets
        
@contextmanager
def openRasterListWithFallback(pathList, **kwargs):
    with ExitStack() as stack:
        datasets = []
        for p in pathList:
            try:
                context = stack.enter_context(rasterio.open(p, **kwargs))
                datasets.append(context)
            except:
                datasets.append(None)
        yield datasets

def _run(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print("Running next:\n", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDERR:\n{completed.stderr.decode(errors='ignore')}"
        )
    return completed

def stackBandsVrt(bandsInput, targetPath):
    """
    Builds a VRT from masked inputs, then applies the overall coverage GeoJSON.
    """
    cmd1 = ["gdalbuildvrt", "-separate", "-resolution", "average", "-r", "cubic", "-overwrite", "-strict"]
    cmd1 += [str(targetPath), *bandsInput]
    _run(cmd1)

    return targetPath