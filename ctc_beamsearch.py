import numpy as np

def joint(k, y, probs, i, Pr, Pr0):
    if k == y[-1]:
        return probs[k] + Pr0[i]
    return probs[k] + Pr[i]

def beamsearch(probs, k=10, blank=0):
    """
    Performs inference for the given output probabilities.

    Arguments:
      `probs`: The output probabilities (e.g. log post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      `k` (int): Size of the beam to use during inference.
      `blank` (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T = probs.shape[0]
    B = [[blank]] # list of prefix
    Pr = [0]
    Pr1 = [-1e30]
    Pr0 = [0]
    for t in range(T):
        # get k most probable sequences in B
        B_new = []
        Pr_new = []
        Pr1_new = []
        Pr0_new = []
        ind = np.argsort(Pr)[::-1]
        B_ = [B[i] for i in ind[:k]]
        for i, y in enumerate(B_):
            if y != [blank]:
                pr1 = Pr1[ind[i]] + probs[t, y[-1]]
                if y[:-1] in B_:
                    pr1 = np.logaddexp(pr1, joint(y[-1], y[:-1], probs[t], B_.index(y[:-1]), Pr, Pr0)) # wrong
            pr0 = Pr[ind[i]] + probs[t, blank]
            if y == [blank]:
                pr1 = -1e30
            B_new += [y]
            Pr_new += [np.logaddexp(pr1, pr0)]
            Pr1_new += [pr1]
            Pr0_new += [pr0]
            for c in range(probs.shape[1]):
                if c == blank: continue
                pr0 = -1e30
                pr1 = joint(c, y, probs[t], ind[i], Pr, Pr0)
                Pr0_new += [pr0]
                Pr1_new += [pr1] 
                Pr_new += [np.logaddexp(pr1, pr0)]
                B_new += [y + [c]]
        B = B_new; Pr = Pr_new; Pr1 = Pr1_new; Pr0 = Pr0_new;
    idx = np.argsort([Pr[i]/len(B[i]) for i in range(len(B))])[::-1]
    # B = [B[i] for i in idx]; Pr = [Pr[i] for i in idx]
    return B[idx[0]], -Pr[idx[0]]

if __name__ == '__main__':
    dist = np.log(np.array([[0.2, 0.1, 0.9],
                             [0.4, 0.4, 0.2],
                             [0.1, 0.3, 0.6], 
                             [0.1, 0.2, 0.3]])) 
    print(beamsearch(dist, k=2))