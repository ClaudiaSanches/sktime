import numpy as np
from sktime.utils.matrix_profile import stomp_lr


def lrstomp(ts, m):
    """
        Computes the left and right matrix profile and index profile.

        Parameters
        ----------
            ts: numpy.array
                Time series.
            m: int
                Length of the subsequences.

        Output
        ------
            lmp: numpy.array
                 Left matrix profile.
            rmp: numpy.array
                 Right matrix profile.
            lip: numpy.array
                 Left index profile.
            rip: numpy.array
                 Right index profile.
    """

    ts_len = ts.shape[0]
    n_subs = ts_len-m+1

    # Initialization
    lmp = np.full(n_subs, float('inf'))
    rmp = np.full(n_subs, float('inf'))

    lip = np.zeros(n_subs)
    rip = np.zeros(n_subs)

    # Computes the
    lmp, lip = stomp_lr(ts, m, 'l')
    rmp, rip = stomp_lr(ts, m, 'r')

    return lmp, rmp, lip, rip


def anchored_chain(lip, rip, j):
    """
        Finds the chain that starts at index j.

        Parameters
        ----------
            lip: numpy.array
                 Left index profile.
            rip: numpy.array
                 Right index profile.
            j: int
               Starting index of the chain.

        Output
        ------
            chain: numpy.array
                   The anchored chain that starts at index j.
    """

    # Initialization
    chain = np.array([j])

    while rip[j] != 0 and lip[int(rip[j])] == j:
        j = int(rip[j])
        chain = np.append(chain, j)

    return chain


def unanchored_chain(ts, m):
    """
        Finds the all-chain set and the unanchored time series chain
        of a time series, and returns both.

        Parameters
        ----------
            ts: numpy.array
                Time series.
            m: int
                Length of the subsequences.

        Output
        ------
            all_chain_set: list
                            List with all the chains that can be obtained
                            from the time series, excluding the ones that
                            are contained in another longer chain.
            unanchored_chain: numpy.array
                              The unanchored chain of the time series.
    """

    n_subs = ts.shape[0] - m+1  # number of subsequences

    # Computes the left and right matrix profile and index profile
    lmp, rmp, lip, rip = lrstomp(ts, m)

    # Initialization
    len_anc_chains = np.ones(n_subs, dtype=int)  # an array that will store the length of the anchored chain starting at each index
    all_chain_set = []

    for i in range(0, n_subs):
        if len_anc_chains[i] == 1:
            j = i
            chains = [j]
            while (rip[j] != -1) and (lip[int(rip[j])] == j):
                j = int(rip[j])
                len_anc_chains[j] = -1
                len_anc_chains[i] = len_anc_chains[i]+1
                chains.append(j)
            all_chain_set.append(chains)

    # Finds the longest chain in the time series
    # If there is more than one chain with the longest length,
    # the tie-breaking criteria used is the chain with the minimum
    # average distance between consecutive components
    max_length = np.amax(len_anc_chains)
    ind = [i for i, j in enumerate(len_anc_chains) if j == max_length]
    if len(ind) > 1:
        chains_same_lenght = np.zeros(len(ind))
        avg_dist = np.zeros(len(ind))
        chains_same_lenght = [anchored_chain(lip, rip, ind[i]) for i in range(0, len(ind))]
        avg_dist = [np.absolute(np.average(np.ediff1d(chains_same_lenght[i]))) for i in range(0, len(ind))]
        unanchored_chain = chains_same_lenght[np.argmin(avg_dist)]
    else:
        unanchored_chain = anchored_chain(lip, rip, ind[0])

    return all_chain_set, unanchored_chain
