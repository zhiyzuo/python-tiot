def read_test_data():
    import csv
    import numpy as np
    from scipy.sparse import lil_matrix

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

    return W, C, vocab, AD, authors, TIMESTAMP

if __name__ == '__main__':
    import sys
    import pickle
    from time import time
    from tiot import GibbsSamplerTIOT
    W, C, vocab, AD, authors, TIMESTAMP = read_test_data()
    #sys.exit(0)

    tiot = GibbsSamplerTIOT(n_iter=200, K=10)
    start = time()
    theta, phi, psi, lambda_ = tiot.fit(W, C, vocab, AD, authors, TIMESTAMP)
    end = time()
    pickle.dump( tiot, open( "tiot.dump", "wb" ) )
    print '---------------'
    print 'Elapsed time: %0.4f seconds' %(end-start)
    print '---------------'
    tiot.show_topics()

