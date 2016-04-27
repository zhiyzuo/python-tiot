class GibbsSamplerTIOT(object):

    def __init__(self, n_iter=1000, burn_in=0.5, K=50, alpha=0.1, beta=0.1, pi=0.1):
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.pi = pi

    def fit(self, W, C, vocab, AD, author_list, timestamp_list, verbose=True):
        # {{{ run gibbs sampling
        import sys
        import numpy as np
        from utils import cartesian
        from scipy.stats import poisson
        '''
            W[:,0] -> word index
            W[:,1] -> document index
            W[:,2] -> timestamp index

            C: document-citation sparse matrix
            C[i, t] -> citation count for the ith document at timestamp t

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

        # save meta-info to this object
        self.vocabulary = vocab
        self.authors = author_list
        self.timestamps = timestamp_list

        # init to one above the max val
        z_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + self.K
        a_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + A
        t_states = np.zeros((self.n_iter+1, nnz), dtype=np.uint32) + T

        # create all needed sequences
        k_range = np.arange(self.K)
        t_range = np.arange(T)
        a_range = np.arange(A)
        v_range = np.arange(V)

        # 1.1 initialize topic assignment randomly
        z_states[0, :] = np.random.choice(self.K, nnz, True)

        # 1.2 initialize author and timestamp assignment
        for i in np.arange(nnz):
            t = W[i, 2]
            di = W[i, 1]
            ad = AD[:, di].nonzero()[0]
            a_states[0, i] = np.random.choice(ad)
            t_states[0, i] = np.random.choice(np.arange(t, T))

        # 2. initialize lambda matrix
        lambda_ = np.zeros((self.K, T), dtype=np.uint32)
        for k in k_range:
            k_indices = np.where(z_states[0, :] == k)[0]
            for t in np.arange(T):
                t_indices = np.where(t_states[0, :] == t)[0]
                kt_indices = np.intersect1d(t_indices, k_indices)
                d_indices = W[kt_indices, 1]
                lambda_[k, t] = C[d_indices, t].mean()

        # zeros set to overall mean
        lam_x, lam_y = np.where(lambda_ == 0)
        if lam_x.size:
            lambda_[lam_x, lam_y] = C.mean()/float((lam_x.size))

        # 3. sample
        # {{{
        for iter_ in np.arange(1, self.n_iter+1):

            if verbose:
                print 'Iter %i...... (Total %i)' %(iter_, self.n_iter)
                sys.stdout.flush()

            else:
                if iter_ % 100 :
                    print 'Iter %i...... (Total %i)' %(iter_, self.n_iter)
                    sys.stdout.flush()
                
            for i in np.arange(nnz):
            #{{{ sample each token sequentially

                # {{{ denominators
                den_author = np.zeros(A, dtype=np.float_)
                for a in a_range:
                    for k in k_range:
                        # words that are assigned to topic k, excluding the current one
                        k_indices = np.append(np.where(z_states[iter_-1, i+1:]==k)[0] + (i+1),\
                                              np.where(z_states[iter_, :i]==k)[0])
                        n_a_k_i = (a_states[iter_-1, k_indices[k_indices > i]] == a).sum() + \
                                  (a_states[iter_, k_indices[k_indices < i]] == a).sum()
                        den_author[a] += n_a_k_i
                    den_author[a] += self.K*self.alpha
                
                den_timestamp = np.zeros(self.K, dtype=np.float_)
                den_token = np.zeros(self.K, dtype=np.float_)
                for k in k_range:

                    for t in t_range:
                        # words that are assigned to timestamp t, excluding the current one
                        t_indices = np.append(np.where(t_states[iter_-1, i+1:]==t)[0] + (i+1),\
                                              np.where(t_states[iter_, :i]==t)[0])
                        n_k_t_i = (z_states[iter_-1, t_indices[t_indices > i]] == k).sum() + \
                                  (z_states[iter_, t_indices[t_indices < i]] == k).sum()
                        den_timestamp[k] += n_k_t_i
                    den_timestamp[k] += T*self.pi

                    for v in v_range:
                        # words that are tokens v, excluding the current one
                        v_indices = np.append(np.where(W[i+1:, 0] == v)[0] + (i+1), np.where(W[:i, 0] == v)[0])
                        n_k_v_i = (z_states[iter_-1, v_indices[v_indices > i]] == k).sum() + \
                                  (z_states[iter_, v_indices[v_indices < i]] == k).sum()
                        den_token[k] += n_k_v_i
                    den_token[k] += V*self.beta

                # }}}

                v = W[i, 0]
                t = W[i, 2]
                di = W[i, 1]
                ci = C[di, :]

                # find its authors
                ad = AD[:,di].nonzero()[0]

                comb_list = cartesian((np.arange(t, T), k_range, ad))
                comb_p_list = np.zeros(comb_list.shape[0], dtype=np.float_)

                # excluding the current one
                v_indices = np.append(np.where(W[i+1:, 0] == v)[0] + (i+1), np.where(W[:i, 0] == v)[0])

                # {{{ for each combination, obtain full conditional probability
                for comb_index in np.arange(comb_p_list.size):

                    comb = comb_list[comb_index]
                    t, k, a = comb

                    # 1
                    t_indices = np.append(np.where(t_states[iter_-1, i+1:]==t)[0] + (i+1),\
                                          np.where(t_states[iter_, :i]==t)[0])
                    n_k_t_i = (z_states[iter_-1, t_indices[t_indices > i]] == k).sum() + \
                              (z_states[iter_, t_indices[t_indices < i]] == k).sum()
                    p1 = (n_k_t_i + self.pi)/den_timestamp[k]

                    # 2
                    n_k_v_i = (z_states[iter_-1, v_indices[v_indices > i]] == k).sum() + \
                              (z_states[iter_, v_indices[v_indices < i]] == k).sum()
                    p2 = (n_k_v_i + self.beta)/den_token[k]

                    # 3
                    # excluding the current one
                    k_indices = np.append(np.where(z_states[iter_-1, i+1:] == k)[0] + (i+1),\
                                          np.where(z_states[iter_, :i] == k)[0])
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
                comb_index = np.random.choice(np.arange(comb_p_list.size), p=comb_p_list)
                t, k, a = comb_list[comb_index]
                t_states[iter_, i] = t
                z_states[iter_, i] = k
                a_states[iter_, i] = a
            #}}} END for i-th TOKEN

            # update lambda after each iteration
            for k in k_range:
                k_indices = np.where(z_states[iter_, :] == k)[0]
                for t in t_range:
                    t_indices = np.where(t_states[iter_, :] == t)[0]
                    kt_indices = np.intersect1d(k_indices, t_indices)
                    d_indices = W[kt_indices, 1]
                    # if no word is assigned to topic k and timestamp t, keep it as before
                    if d_indices.size > 0:
                        lambda_[k, t] = C[d_indices, t].mean()
        # }}}

        # 4. obtain \theta, \phi, and \psi

        # burn-in: first half
        z_samples = z_states[1:, :][self.n_iter/2:, :]
        a_samples = a_states[1:, :][self.n_iter/2:, :]
        t_samples = t_states[1:, :][self.n_iter/2:, :]

        # author-topic
        theta = np.zeros((A, self.K), dtype=np.float_)
        # topic-word
        phi = np.zeros((self.K, V), dtype=np.float_)
        # topic-timestamp
        psi = np.zeros((self.K, T), dtype=np.float_)

        for a in a_range:
            den = self.K * self.alpha + (a_samples==a).sum()
            a_x, a_y = np.where(a_samples==a)
            for k in k_range:
                n_a_k = (z_samples[a_x, a_y] == k).sum()
                theta[a,k] = float(n_a_k+self.alpha) / (den)

        for k in k_range:
            k_count = (z_samples==k).sum()
            den_v = V * self.beta + k_count
            den_t = T * self.pi + k_count
            # x is iteration number, y is word index
            k_x, k_y = np.where(z_samples==k)
            for v in v_range:
                n_k_v = (W[k_y, 0]==v).sum()
                phi[k,v] = float(n_k_v+self.beta) / den_v

            for t in t_range:
                n_k_t = (t_samples[k_x, k_y]==t).sum()
                psi[k,t] = float(n_k_t+self.pi) / den_t

                # update lambda
                t_x, t_y = np.where(t_samples == t)[0]
                kt_indices = np.intersect1d(k_y, t_y)
                d_indices = W[kt_indices, 1]
                # if no word is assigned to topic k and timestamp t, keep it as before
                if d_indices.size > 0:
                    lambda_[k, t] = C[d_indices, t].mean()

        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.lambda_ = lambda_
        self.z_samples = z_samples
        self.a_samples = a_samples
        self.t_samples = t_samples
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

        topic_words = self.vocabulary[np.argsort(-self.phi)[:, :top_n_words]]

        for k in np.argsort(-self.topic_proportion):
            print 'Topic %i (%0.4f): %s' %(k, self.topic_proportion[k], ', '.join(topic_words[k]))

    def show_author_topics(self, top_n=None):
        import numpy as np

        if top_n is None:
            top_n = self.K
        
        A = self.theta.shape[0]
        top_topic_indices = np.argsort(-self.theta)[:, :top_n]
        for a in np.arange(A):
            print '%s: %s' %(self.authors[a], ', '.join(top_topic_indices[a, :].astype(str)))

    def show_topic_timestamps(self, top_t=None):
        import numpy as np

        T = self.timestamps.size
        if top_t is None:
            top_t = T

        top_timestamps = self.timestamps[np.argsort(-self.psi)[:, :top_t]]
        for k in np.arange(self.K):
            print 'Topic %i: %s' %(k, ', '.join(top_timestamps[k, :].astype(str)))

    def show_topic_impact(self, top_t=None):
        import numpy as np

        T = self.timestamps.size
        if top_t is None:
            top_t = T

        top_timestamps = np.argsort(-self.lambda_)[:, :top_t]
        print '****** Topical Impact by Citation ******'
        for k in np.arange(self.K):
            print 'Topic %i: %s' %(k, ', '.join(top_timestamps[k,:].astype(str)))

class Preprocessor(object):

    def __init__(self):
        self.corpus = bow
        self.citation = citation_list
        self.timestamp = timestamp_list

