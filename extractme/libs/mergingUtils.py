from tqdm import tqdm

def sequentialSum(sequence, progress=True):
    sequence = list(sequence)
    res = sequence[0]
    for s in tqdm(sequence[1:], desc=f"Sequential summation of {type(res)} objects", disable=not progress):
        res = res+s
    return res

def pairwiseSum(sequence, progress=True):
    sequence = list(sequence)
    
    if progress:
        print(f"Pairwise summation of {type(sequence[0])} objects")
    step=0
    while len(sequence) > 1:
        step += 1
        # Sum adjacent pairs
        iterator = range(0, len(sequence)-1, 2)
        if progress:
            iterator = tqdm(iterator, desc=f"Step {step} pairs", leave=False)
        sequence = [sequence[i] + sequence[i+1] for i in iterator] + ([sequence[-1]] if len(sequence) % 2 else [])
    return sequence[0]

def minorizationSum(sequence, weights=None, progress=True):
    seq = list(sequence)
    if len(seq)==0:
        raise ValueError("sequence must contain at least one element")

    if weights is None:
        w = [1.0] * len(seq)
    else:
        w = list(weights)
        if len(w) != len(seq):
            raise ValueError("weights must have same length as sequence")

    # Represent current items as list of (element, weight)
    items = list(zip(seq, w))

    if progress:
        print(f"Minorization summation of {type(seq[0])} objects")
    step = 0
    while len(items) > 1:
        step += 1

        # If odd length: choose index to skip (closest to future mean weight)
        skipped = None
        if len(items) % 2 == 1:
            mean_w = sum(it[1] for it in items) / len(items)*2
            idx_to_skip = max(
                range(len(items)),
                key=lambda i: (abs(items[i][1] - mean_w), items[i][1])
            )
            skipped = items.pop(idx_to_skip)

        # sort by weight ascending so we pair smallest with largest
        items.sort(key=lambda it: it[1])

        new_items = []
        pair_count = len(items) // 2
        iterator = range(pair_count)
        if progress:
            iterator = tqdm(iterator, desc=f"Step {step} pairs", leave=False)

        for i in iterator:
            low_elem, low_w = items[i]
            high_elem, high_w = items[-1 - i]
            # sum elements and weights
            new_elem = low_elem + high_elem
            new_weight = low_w + high_w
            new_items.append((new_elem, new_weight))

        # re-insert skipped element (if any) to be processed in next round
        if skipped is not None:
            new_items.append(skipped)

        items = new_items

    # items[0] is (element, weight); return element only to match original API
    return items[0][0]