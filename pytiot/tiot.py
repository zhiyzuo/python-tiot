class GibbsSamplerTIOT(object):

    def __init__(self, n_iter=1000, burn_in=0.5, K=50, alpha=0.1, beta=0.1, pi=0.1):
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.pi = pi

    def fit(self, W, C, vocab, AD, author_list, timestamp_list):
        # {{{ run gibbs sampling
        import numpy as np
        from utils import cartesian
        from scipy.stats import poisson
        from progress.bar import Bar
        '''
            W[:,0] -> word index
            W[:,1] -> document index
            W[:,2] -> timestamp index

            C: token-citation sparse matrix
            C[i, t] -> citation count for the ith token at timestamp t

            AD: author-document sparse matrix
        '''

        # number of authors
        A = author_list.size
        # number of unique tokens
        V = vocab.size
        # total number of tokens
        nnz = W.shape[0]
        # number of timestamps
        T = timestamp_list.size

        self.vocabulary = vocab

        # \theta, \phi, \psi
        theta = np.zeros((A, self.K), dtype=np.float_)
        phi = np.zeros((self.K, V), dtype=np.float_)
        psi = np.zeros((self.K, T), dtype=np.float_)

        # init all states to one above the max val
        z_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + self.K + 1
        a_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + A + 1
        t_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + T + 1
        c_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32)

        # 1.1 initialize topic assignment
        z_states[0, :] = np.random.choice(self.K, nnz)
        t_states[0, :] = np.random.choice(T, nnz)
        # 1.2 initialize author assignment
        for i in np.arange(nnz):
            di = W[i, 1]
            ti = W[i, 2]
            ad = AD[:, di].nonzero()[0]
            a_states[0, i] = np.random.choice(ad)
            c_states[0, i] = np.random.poisson(np.mean(C[di,:]))

        # 2. initialize lambda matrix: avg. citation for words
        avg_citation = np.mean(C)
        lambda_ = np.random.poisson(avg_citation, size=(self.K, T))
        
        # 3. sample
        # {{{
        for iter_ in np.arange(1, self.n_iter+1):
            print 'Iter %i' %iter_
            bar = Bar('Processing', max=nnz)
            for i in np.arange(nnz):

                # {{{ denominators
                den_author = np.zeros(A, dtype=np.float_)
                for a in np.arange(A):
                    for k in np.arange(self.K):
                        # words that are assigned to topic k, excluding the current one
                        k_indices = np.append(np.where(z_states[iter_-1, i+1:]==k)[0] + (i+1),\
                                              np.where(z_states[iter_, :i]==k)[0])
                        n_a_k_i = (a_states[iter_-1, k_indices[k_indices > i]] == a).sum() + \
                                  (a_states[iter_, k_indices[k_indices < i]] == a).sum()
                        den_author[a] += n_a_k_i
                    den_author[a] += self.K*self.alpha
                
                den_timestamp = np.zeros(self.K, dtype=np.float_)
                den_token = np.zeros(self.K, dtype=np.float_)
                for k in np.arange(self.K):
                    for t in np.arange(T):
                        # words that are assigned to timestamp t, excluding the current one
                        t_indices = np.append(np.where(t_states[iter_-1, i+1:]==t)[0] + (i+1),\
                                              np.where(t_states[iter_, :i]==t)[0])
                        n_k_t_i = (z_states[iter_-1, t_indices[t_indices > i]] == k).sum() + \
                                  (z_states[iter_, t_indices[t_indices < i]] == k).sum()
                        den_timestamp[k] += n_k_t_i
                    den_timestamp[k] += T*self.pi

                    for v in np.arange(V):
                        # words that are tokens v, excluding the current one
                        v_indices = np.append(np.where(W[i+1:, 0]==v)[0] + (i+1),\
                                              np.where(W[:i, 0]==v)[0])
                        n_k_v_i = (z_states[iter_-1, v_indices[v_indices > i]] == k).sum() + \
                                  (z_states[iter_, v_indices[v_indices < i]] == k).sum()
                        den_token[k] += n_k_v_i
                    den_token[k] += V*self.beta

                # }}}

                v = W[i, 0]
                di = W[i, 1]
                ci = C[di, :]

                # find its authors
                ad = AD[:,di].nonzero()[0]

                comb_list = cartesian((np.arange(T), np.arange(self.K), ad))
                comb_p_list = np.zeros(comb_list.shape[0], dtype=np.float_)
                # {{{ for each combination, obtain full conditional probability
                for comb_index in np.arange(comb_p_list.size):

                    comb = comb_list[comb_index]
                    t, k, a = comb

                    # find timestamp ti's topic assignments
                    t_indices = np.append(np.where(t_states[iter_-1, i+1:]==t)[0] + (i+1),\
                                          np.where(t_states[iter_, :i]==t)[0])
                    n_k_t_i = (z_states[iter_-1, t_indices[t_indices > i]] == k).sum() + \
                              (z_states[iter_, t_indices[t_indices < i]] == k).sum()
                    p1 = (n_k_t_i + self.pi)/den_timestamp[k]

                    # v is just wi
                    v_indices = np.append(np.where(W[i+1:, 0]==v)[0] + (i+1),\
                                          np.where(W[:i, 0]==v)[0])
                    n_k_v_i = (z_states[iter_-1, v_indices[v_indices > i]] == k).sum() + \
                              (z_states[iter_, v_indices[v_indices < i]] == k).sum()
                    p2 = (n_k_v_i + self.beta)/den_token[k]

                    # words that are assigned to topic k
                    k_indices = np.append(np.where(z_states[iter_-1, i+1:]==k)[0] + (i+1),\
                                          np.where(z_states[iter_, :i]==k)[0])
                    n_a_k_i = (a_states[iter_-1, k_indices[k_indices > i]] == a).sum() + \
                              (a_states[iter_, k_indices[k_indices < i]] == a).sum()
                    p3 = (n_a_k_i + self.alpha)/den_author[a]

                    # poisson pmf
                    p4 = poisson.pmf(ci[t], mu=lambda_[k,t])

                    #print p1, p2, p3, p4, lambda_[k,t], ci[t]
                    comb_p_list[comb_index] = p1*p2*p3*p4
                # }}}
                
                # rescale to [0,1]
                comb_p_list = comb_p_list/comb_p_list.sum()

                # sample for i-th word
                comb_index = np.random.choice(np.arange(comb_list.shape[0]), replace=False, p=comb_p_list)
                t, k, a = comb_list[comb_index]
                z_states[iter_, i] = k
                a_states[iter_, i] = a
                t_states[iter_, i] = t
                c_states[iter_, i] = np.random.poisson(lambda_[k, t])

                bar.next()
            bar.finish()

            # update lambda
            for k in np.arange(self.K):
                for t in np.arange(T):
                    k_indices = np.where(z_states[iter_, :] == k)[0]
                    t_indices = np.where(t_states[iter_, :] == t)[0]
                    kt_indices = np.intersect1d(k_indices, t_indices)
                    # if no word is assigned to topic k and timestamp t, keep it as before
                    try:
                        lambda_[k,t] = 1./kt_indices.size * c_states[iter_, kt_indices].sum()
                    except:
                        continue
        # }}}

        # 4. update \theta, \phi, and \psi
        # burn-in: first half
        z_samples = z_states[self.n_iter/2:, :]
        a_samples = a_states[self.n_iter/2:, :]
        t_samples = t_states[self.n_iter/2:, :]

        for a in np.arange(A):
            den = self.K * self.alpha + (a_samples==a).sum()
            a_x, a_y = np.where(a_samples==a)
            for k in np.arange(self.K):
                n_a_k = (z_samples[a_x, a_y] == k).sum()
                theta[a,k] = float(n_a_k+self.alpha) / (den)

        for k in np.arange(self.K):
            k_count = (z_samples==k).sum()
            den_v = V * self.beta + k_count
            den_t = T * self.pi + k_count
            k_x, k_y = np.where(z_samples==k)
            for v in np.arange(V):
                n_k_v = (W[k_y, 0]==v).sum()
                phi[k,v] = float(n_k_v+self.beta) / den_v

            for t in np.arange(T):
                n_k_t = (t_samples[k_x,k_y]==t).sum()
                psi[k,t] = float(n_k_t+self.pi) / den_t

        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.lambda_ = lambda_
        self.z_samples = z_samples
        print self.z_samples
        self.t_samples = t_samples
        self.a_samples = a_samples
        # }}}
        return theta, phi, psi, lambda_

    def show_topics(self, top_n_words=10, top_n=None):
        # print top-n topics
        import numpy as np
        from scipy.stats import itemfreq

        if top_n is None:
            top_n = self.K

        # get topic proportion
        topic_occurences = itemfreq(self.z_samples)
        self.topic_proportion = topic_occurences[:,1] / topic_occurences[:,1].astype(np.float).sum()

        topic_words = []
        for k in np.arange(self.K):
            top_word_indices = np.argsort(-self.phi[k])[-top_n_words:]
            topic_words.append(self.vocabulary[top_word_indices])

        for k in np.argsort(-self.topic_proportion):
            print 'Topic %i (%0.4f): %s' %(k, self.topic_proportion[k], ', '.join(topic_words[k]))

    def show_author_topics(self):
        import numpy as np
        pass


class Preprocessor(object):

    def __init__(self):
        self.corpus = bow
        self.citation = citation_list
        self.timestamp = timestamp_list
