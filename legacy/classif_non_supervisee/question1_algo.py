import numpy as np

class Bernouilli_EM:
    k = 0
    d = 0
    n = 0
    h = None
    p = None
    pi = None
    seed = None
    e = 0.000001
    convergence_threshold = 0.1

    def __init__(self, n_clusters=None, random_state=None):
        self.k = n_clusters
        self.seed = random_state

    def fit(self, data):
        self.d = data.shape[1]
        self.n = data.shape[0]
        self.h = np.zeros((self.n, self.k))
        self.p = np.zeros((self.k, self.d))
        self.pi = np.zeros(self.k)

        self.initialise_through_k_means(data)


        for i in range(0, 100):
            previous_h = self.h
            previous_p = self.p
            previous_pi = self.pi
            self.h = self.e_step(data)
            self.p, self.pi = self.m_step(data)

            errors = np.array([(np.abs(previous_h - self.h)).sum(),
                               (np.abs(previous_pi - self.pi)).sum(),
                               (np.abs(previous_pi - self.pi)).sum()])
            print(errors)
            if errors.sum() <= self.convergence_threshold:
                break

    def initialise_through_k_means(self, data):
        if self.seed is not None:
            np.random.seed(self.seed)
        indices_range = np.arange(0,self.n)
        np.random.shuffle(indices_range)

        m = data[indices_range[:self.k], :]

        distances = np.linalg.norm(data[:, np.newaxis] - m, axis=2)

        self.h = np.zeros((self.n, self.k))
        closest_center_index = np.argmin(distances, axis=1)
        for t in range(0, self.n):
            self.h[t, closest_center_index[t]] = 1

        self.h[self.h < self.e] = self.e

        self.p, self.pi = self.m_step(data)

    def e_step(self, data):
        h = np.prod(np.power(self.p, data[:, np.newaxis]) * np.power((1 - self.p), (1 - data[:, np.newaxis])), axis=2) * self.pi
        h = h / np.sum(h, axis=1)[:, None]
        return h

    def m_step(self, data):
        #h_divided_by_column_mean = self.h/self.h.sum(axis=0)[:, None].T
        #p = h_divided_by_column_mean.T.dot(data)
        #print(np.sum(self.h, axis=0))
        p = ((self.h.T.dot(data)).T / np.sum(self.h, axis=0)).T

        #p = np.zeros((self.k, self.d))
        #for i in range(0, self.k):
        #    for j in range(0, self.d):
        #        sum_hitxjt = 0
        #        sum_hit = 0
        #        for t in range(0, self.n):
        #            sum_hitxjt += self.h[t, i] * data[t, j]
        #            sum_hit += self.h[t, i]
        #        p[i, j] = sum_hitxjt/sum_hit

        p[p < self.e] = self.e
       # print(p)

        pi = np.mean(self.h, axis=0)

        pi[pi < self.e] = self.e

        return p, pi

    def predict(self, data):
        h = self.e_step(data)
        z = np.argmax(h, axis=1)
        return z

    def score(self, data):
        h = self.e_step(data)
        score = 0
        for j in range(0, self.k):
            for t in range(0, self.n):
                log_p_x = 0
                for i in range(0, self.d):
                    log_p_x += data[t, i] * np.log(self.p[j, i]) + (1 - data[t, i]) * np.log(1 - self.p[j,i])
                score += h[t, j] * np.log(self.pi[j]) + h[t, j] * log_p_x
        return score
