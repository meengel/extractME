import numpy as np
from .ERADist import ERADist

# median related
def median(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = np.multiply(coverage, weights)
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.nanquantile(values, 0.5, weights=normalizedWeights, method="inverted_cdf")

def unweightedMedian(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = coverage
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.nanquantile(values, 0.5, weights=normalizedWeights, method="inverted_cdf")

# mean related
def mean(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = np.multiply(coverage, weights)
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.nansum(np.multiply(values, normalizedWeights))

def unweightedMean(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = coverage
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.nansum(np.multiply(values, normalizedWeights))

# var related
def var(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = np.multiply(coverage, weights)
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.cov(values, aweights=normalizedWeights)

def unweightedVar(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    preWeights = coverage
    normalizedWeights = preWeights/np.sum(preWeights)
    return np.cov(values, aweights=normalizedWeights)

# std related
def std(values, coverage, weights, ignoreValues=[], filterFun=None):
    return np.sqrt(var(values, coverage, weights, ignoreValues=ignoreValues, filterFun=filterFun))

def unweightedStd(values, coverage, weights, ignoreValues=[], filterFun=None):
    return np.sqrt(unweightedVar(values, coverage, weights, ignoreValues=ignoreValues, filterFun=filterFun))

# weight related
def weight(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    return np.sum(coverage*weights)

def unweightedWeight(values, coverage, weights, ignoreValues=[], filterFun=None):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    return np.sum(coverage)
    
# filter and masking
def filterValid(values, coverage, weights, ignoreValues=[]):
    valid = np.logical_and(~np.isnan(values),~np.isin(values, ignoreValues))
    values = np.asarray(values)[valid]
    coverage = np.asarray(coverage)[valid]
    weights = np.asarray(weights)[valid]
    return values, coverage, weights

def filterAllTouched(values, coverage, weights):
    filteredIdx = coverage!=0
    return values[filteredIdx], filteredIdx[filteredIdx].astype(float), weights[filteredIdx]

def filterFullyContained(values, coverage, weights):
    filteredIdx = coverage==1
    return values[filteredIdx], coverage[filteredIdx], weights[filteredIdx]

def filterPartiallyContained(values, coverage, weights):
    filteredIdx = np.logical_and(coverage!=0, coverage!=1)
    return values[filteredIdx], coverage[filteredIdx], weights[filteredIdx]

def filterCoverageInterval(values, coverage, weights, minCoverage=0, maxCoverage=1):
    filteredIdx = np.logical_and(coverage>=minCoverage, coverage<=maxCoverage)
    return values[filteredIdx], coverage[filteredIdx], weights[filteredIdx]

def filterBoundary(values, coverage, weights):
    filteredIdx = np.logical_and(coverage!=0, coverage!=1)
    return values[filteredIdx], filteredIdx[filteredIdx].astype(float), weights[filteredIdx]

# quantile related
def make_quantile_fn(q, ignoreValues=[], filterFun=None):
    def quantile(values, coverage, weights=None, ignoreValues=ignoreValues, filterFun=filterFun):
        values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
        if filterFun is not None:
            values, coverage, weights = filterFun(values, coverage, weights)
        if len(values)==0 or np.sum(coverage)==0:
            return None
        return np.nanquantile(values, q, weights=coverage*weights/np.sum(coverage*weights), method="inverted_cdf")
    quantile.__name__ = f"quantile-{str(q).replace('.', 'p')}"
    return quantile

def make_unweightedQuantile_fn(q, ignoreValues=[], filterFun=None):
    def unweightedQuantile(values, coverage, weights=None, ignoreValues=ignoreValues):
        values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
        if filterFun is not None:
            values, coverage, weights = filterFun(values, coverage, weights)
        if len(values)==0 or np.sum(coverage)==0:
            return None
        return np.nanquantile(values, q, weights=coverage/np.sum(coverage), method="inverted_cdf")
    unweightedQuantile.__name__ = f"unweightedQuantile-{str(q).replace('.', 'p')}"
    return unweightedQuantile

# empirical distribution
def buildEmpDist(values, coverage, weights, ignoreValues=[], filterFun=None, doubleOneSampled=True, pdfMethod="nearest", pdfPoints=None, **pdfMethodParams):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    if len(values)<2:
        if doubleOneSampled:
            values = np.concat([values, values])
            coverage = np.concat([coverage, coverage])
            weights = np.concat([weights, weights])
        else:
            return None
    
    return ERADist(
        'empirical',
        'DATA',
        [values, coverage*weights, pdfMethod, pdfPoints, pdfMethodParams]
    )

def buildUnweightedEmpDist(values, coverage, weights, ignoreValues=[], filterFun=None, doubleOneSampled=True, pdfMethod="nearest", pdfPoints=None, **pdfMethodParams):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    if len(values)<2:
        if doubleOneSampled:
            values = np.concat([values, values])
            coverage = np.concat([coverage, coverage])
        else:
            return None
        
    return ERADist(
        'empirical',
        'DATA',
        [values, coverage, pdfMethod, pdfPoints, pdfMethodParams]
    )

# empirical digest
def buildEmpDigest(values, coverage, weights, ignoreValues=[], filterFun=None, doubleOneSampled=True, nApprox=1000, mode="densityFocused", eps=None, pdfMethod="nearest", pdfPoints=None, **pdfMethodParams):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    if len(values)<2:
        if doubleOneSampled:
            values = np.concat([values, values])
            coverage = np.concat([coverage, coverage])
            weights = np.concat([weights, weights])
        else:
            return None
        
    return ERADist(
        'empiricaldigest',
        'DATA',
        [values, coverage*weights, nApprox, mode, eps, pdfMethod, pdfPoints, pdfMethodParams]
    )

def buildUnweightedEmpDigest(values, coverage, weights, ignoreValues=[], filterFun=None, doubleOneSampled=True, nApprox=1000, mode="densityFocused", eps=None, pdfMethod="nearest", pdfPoints=None, **pdfMethodParams):
    values, coverage, weights = filterValid(values, coverage, weights, ignoreValues=ignoreValues)
    if filterFun is not None:
        values, coverage, weights = filterFun(values, coverage, weights)
    if len(values)==0 or np.sum(coverage)==0:
        return None
    
    if len(values)<2:
        if doubleOneSampled:
            values = np.concat([values, values])
            coverage = np.concat([coverage, coverage])
        else:
            return None
        
    return ERADist(
        'empiricaldigest',
        'DATA',
        [values, coverage, nApprox, mode, eps, pdfMethod, pdfPoints, pdfMethodParams]
    )

