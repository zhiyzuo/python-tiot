class GibbsSamplerTIOT(object):

    def __init__(self, n_iter=1000, burn_in=0.5, K=50, alpha=0.1, beta=0.1, pi=0.1):
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.pi = pi

    def fit(W, AD, author_list=None, timestamp_list=None):
        from utils import cartesian
        from scipy.stats import poisson
        from scipy.special import gamma
        '''
            W[:,0] -> word index
            W[:,1] -> document index
            W[:,2] -> timestamp index
            W[:,3] -> citation count

            AD: author-document sparse matrix
        '''

        # number of authors
        A = AD.shape[0]
        # total number of tokens
        nnz = np.unique(W[:,0]).size
        # number of timestamps
        T = np.unique(W[:,2]).size

        z_states = np.empty((self.n_iter, V), dtype=np.uint32)
        a_states = np.empty((self.n_iter, V), dtype=np.uint32)
        t_states = np.empty((self.n_iter, V), dtype=np.uint32)

        # 1.1 initialize topic assignment
        z_states[0, :] = np.random.choice(self.K, V)
        # 1.2 initialize author assignment
        for i in np.arange(nnz):
            di = [Wi, 1]
            ad = AD[:, di].nonzero()[0]
            a_states[0, i] = np.random.choice(ad)

        # 2. initialize lambda matrix: avg. citation for all documents
        #TODO
        avg_citation = 20
        lambda_ = np.empty((self.K, T), dtype=np.uint32)

        # 3. sample
        for iter_ in np.arange(1, self.n_iter):
            for i in np.arange(nnz):

                # {{{ denominators
                den_author = np.empty((1,A), dtype=float)
                for a in np.arange(A):
                    for k in np.arange(self.K):
                        # words that are assigned to topic k
                        k_indices = np.where(z_states[iter_-1,:]==k)[0]
                        n_a_k = (a_states[iter_-1, k_indices] == a).sum()
                        den_author[1, a] += gamma(n_a_k - 1 + self.K*self.alpha)

                den_timestamp = np.empty((1,self.K), dtype=float)
                for k in np.arange(self.K):
                    for t in np.arange(T):
                        t_indices = np.where(W[:, 2]==t)[0]
                        t_topic_assignments = z_states[iter_-1, t_indices]
                        n_k_t = (t_topic_assignments == k).sum()
                        den_timestamp[1, k] += gamma(n_k_t - 1 + T*self.pi)

                den_token = np.empty((1,self.K), dtype=float)
                for k in np.arange(self.K):
                    for v in np.arange(V):
                        v_indices = np.where(W[:, 0]==v)[0]
                        v_topic_assignments = z_states[iter_-1, v_indices]
                        n_k_v = (v_topic_assignments == k).sum()
                        den_token[1, k] += gamma(n_k_v - 1 + V*self.beta)
                # }}}

                # find its belonging doc, word, timestamp index, and citation count
                di = W[i, 1]
                wi = W[i, 0]
                ti = W[i, 2]
                ci = W[i, 3]
                # find its authors
                ad = AD[:,di].nonzero()[0]

                # find token wi's topic assignments
                wi_indices = np.where(W[:, 0]==wi)[0]
                wi_topic_assignments = z_states[iter_-1, wi_indices]

                comb_list = cartesian((np.arange(T), np.arange(self.K), ad))
                comb_p_list = np.zeros(comb_list.shape[0])
                #{{{ for each combination, obtain full conditional probability
                for comb_index in np.arange(comb_p_list.size):

                    comb = comb_list[comb_index]
                    t, k, a = comb

                    # find timestamp ti's topic assignments
                    t_indices = np.where(W[:, 2]==t)[0]
                    n_k_t = (z_states[iter_-1, t_indices]==k).sum()
                    p1 = gamma(n_k_t - 1 + self.pi)/den_timestamp[1,k]

                    # v is just wi
                    n_k_v = (wi_topic_assignments==k).sum()
                    p2 = gamma(n_k_v - 1 + self.beta)/den_token[1,k]

                    # words that are assigned to topic k
                    k_indices = np.where(z_states[iter_-1,:]==k)[0]
                    n_a_k = (a_states[iter_-1, k_indices] == a).sum()
                    p3 = gamma(n_a_k - 1 + self.alpha)/den_author[1,a]

                    # poisson pmf
                    # lambda_ matrix TODO
                    p4 = poisson.pmf(ci, mu=lambda_[k,t])

                    comb_p_list[comb_index] = p1*p2*p3*p4
                # }}}
                
                # rescale to [0,1]
                comb_p_list /= comb_p_list.sum()

                # sample for i-th word
                t, k, a = np.random.choice(comp_list, size=1, False, p=comb_p_list)
                z_states[iter_, i] = k
                a_states[iter_, i] = a
             
