from .zonalExtractor import extractChosenOps, extractQuantity, processRawExtractionToQuantity, processRawExtraction
from .areaCalculator import calculatePixelAreaTif
from .libs.utils import stringifyGdf, destringifyGdf
from .libs.ERADist import ERADist
from .libs.ERANataf import ERANataf
from .libs.EmpiricalDist import EmpDist, EmpDigest

_EMPDISTOPS = [
    # empDist operations
    "empDist",
    "allTouchedEmpDist",
    "boundaryEmpDist",
    "fullyContainedEmpDist",
    "partiallyContainedEmpDist",
    "unweightedEmpDist",
    "unweightedAllTouchedEmpDist",
    "unweightedBoundaryEmpDist",
    "unweightedFullyContainedEmpDist",
    "unweightedPartiallyContainedEmpDist"] + [
    # centroidBasedEmpDist operations
    "centroidBasedEmpDist",
    "centroidBasedBoundaryEmpDist",
    "centroidBasedFullyContainedEmpDist",
    "unweightedCentroidBasedEmpDist",
    "unweightedCentroidBasedBoundaryEmpDist",
    "unweightedCentroidBasedFullyContainedEmpDist"] + [
    # empDigest operations
    "empDigest",
    "allTouchedEmpDigest",
    "boundaryEmpDigest",
    "fullyContainedEmpDigest",
    "partiallyContainedEmpDigest",
    "unweightedEmpDigest",
    "unweightedAllTouchedEmpDigest",
    "unweightedBoundaryEmpDigest",
    "unweightedFullyContainedEmpDigest",
    "unweightedPartiallyContainedEmpDigest"] + [
    # centroidBasedEmpDigest operations
    "centroidBasedEmpDigest",
    "centroidBasedBoundaryEmpDigest",
    "centroidBasedFullyContainedEmpDigest",
    "unweightedCentroidBasedEmpDigest",
    "unweightedCentroidBasedBoundaryEmpDigest",
    "unweightedCentroidBasedFullyContainedEmpDigest"
]

__all__ = [
    "extractChosenOps",
    "extractQuantity",
    "processRawExtractionToQuantity",
    "processRawExtraction",
    "calculatePixelAreaTif",
    "stringifyGdf",
    "destringifyGdf",
    "ERADist",
    "ERANataf",
    "EmpDist",
    "EmpDigest",
]