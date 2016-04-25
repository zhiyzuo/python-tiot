import csv
import numpy as np
from scipy.sparse import lil_matrix

def read_test_data():

    with open('./test_data.csv', 'rb') as f:
        reader = csv.reader(f)
        data = [line for line in reader]

    header = np.asarray(data[0])
    ids = np.asarray(sorted([item[0] for item in data[1:]]))
    d = {item[0]:item[1:] for item in data[1:]}

    ### citation-timestamp matrix ###
    ### timestamp list ###

    TIMESTAMP = np.arange(2010, 2016)
    C = np.empty((ids.size, TIMESTAMP.size), dtype=np.uint32)
    for index in np.arange(ids.size):
        # citation #
        C[index, :] = d[ids[index]][3:-2]

    ### author-doc matrix ###
    with open('./test_author.csv', 'rb') as f:
        reader = csv.reader(f)
        data = [line for line in reader]

    doc_author_dict = {item[0]:np.asarray([x for x in item[1:] if len(x)>0]) for item in data}
    authors = np.unique([x for item in doc_author_dict.values() for x in item])

    AD = lil_matrix((authors.size, ids.size), dtype=np.uint32)
    for index in np.arange(ids.size):
        id_ = ids[index] 
        doc_authors = doc_author_dict[id_]
        indices = np.where(np.in1d(authors, doc_authors))[0]
        AD[indices, index] = 1

    ### bag of words ###

    '''
    import string
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    ps = PorterStemmer()
    texts = np.array([d[ids[index]][1] + ' ' + d[ids[index]][0] for index in np.arange(ids.size)])
    stop = stopwords.words('english') + list(string.punctuation)

    bow = [np.unique([ps.stem(item) for item in word_tokenize(te.lower()) if item not in stop and len(item) > 2]) for te in texts]
    nnz = sum([len(item) for item in bow])

    vocab = np.unique(np.array([v for doc in bow for v in doc]))

    W = np.empty((nnz, 2), dtype=np.uint32)
    index = 0
    for i in np.arange(ids.size): 
        for j in np.arange(len(bow[i])):
            w_index = np.where(vocab==bow[i][j])[0]
            # token index
            W[index,0] = w_index
            # doc index
            W[index,1] = i

            index += 1
    '''

    return C, AD, authors, TIMESTAMP

def mat2wd(matfile):

    num_doc = int(data[0][0])
    num_word = int(data[0][1])
    nnz = int(data[0][2])

    WD = np.zeros((num_word, num_doc))
    matf = matfile[1:]

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

if __name__ == '__main__':
    import pickle
    import sys, csv
    import numpy as np
    from time import time
    from tiot import GibbsSamplerTIOT
    C, AD, authors, TIMESTAMP = read_test_data()

    with open('test_mat.txt', 'r') as f:
        reader = csv.reader(f)
        data = [line[0].split() for line in reader]
    
    WD = mat2wd(data)
    nnz, W = sparse_mat_count(WD)
    #sys.exit(0)

    with open('test_mat.txt.clabel', 'r') as f:
        reader = csv.reader(f)
        vocab = np.asarray([line[0] for line in reader])

    tiot = GibbsSamplerTIOT(n_iter=500, K=10)
    start = time()
    theta, phi, psi, lambda_ = tiot.fit(W, C, vocab, AD, authors, TIMESTAMP)
    end = time()
    pickle.dump( tiot, open( "tiot.dump", "wb" ) )
    print '---------------'
    print 'Elapsed time: %0.4f seconds' %(end-start)
    print '---------------'
    tiot.show_topics()

