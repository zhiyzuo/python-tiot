import numpy as np

def mat2wd(data):

    num_doc = int(data[0][0])
    num_word = int(data[0][1])
    nnz = int(data[0][2])

    WD = np.zeros((num_word, num_doc))
    matf = data[1:]

    for i in np.arange(num_doc):
        l = map(int, matf[i])
        for j in np.arange(start=0, stop=len(l), step=2):
            index, occurrence = l[j], l[j+1]
            WD[index-1, i] = occurrence
    return WD

def sparse_mat_count(WD):
    nnz = int(WD.sum())
    W = np.zeros((nnz, 2), dtype=np.int)
    nnz_x, nnz_y = np.where(WD!=0)
    start_index = 0
    for i in np.arange(nnz_x.size):
        word_occurence = int(WD[nnz_x[i], nnz_y[i]])
        W[start_index:(start_index+word_occurence), 0] = int(nnz_x[i])
        W[start_index:(start_index+word_occurence), 1] = int(nnz_y[i])
        start_index += word_occurence
    return nnz, W

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
