import numpy as np

def znorm(ts, m):
    """
        Z-normalizes a time series.

        Parameters
        ----------
            ts: pandas.Series
                Time series.
            m: int
                Time series' length.

        Output
        ------
            znorm_ts: panda.Series
                Z-normalized time series.
    """

    ts_mean = np.mean(ts) # mean
    ts_std = np.std(ts) # standard deviation

    znorm_ts = (ts[0:m]-ts_mean)/ts_std

    return znorm_ts

def power_sum(subs, m):

    """
        Calculates the sum of the power of two of all the elements from a subsequence.

        Parameters
        ----------
            subs: pandas.Series
                Subsequence.
            m: int
                Length of the subsequence.

        Output
        ------
            s: float
                Sum of the power of two of all the elements from subs.

    """

    s = 0
    for i in range(0, m):
        s += (pow(subs[i],2))

    return s

def stomp(ts1, ts2, m):

    """
        STOMP (Scalable Time series Ordered-search Matrix Profile) implementation.

        Parameters
        ----------
            ts1: pandas.Series
                First time series.
            ts2: pandas.Series
                Second time series.
            m: int
                Length of the subsequences.

        Output
        ------
            MP: list
                List with the distance between every subsequence from ts1 to the nearest subsequence with same length from ts2.
    """

    # Length of ts1 and ts2
    len1 = ts1.size
    len2 = ts2.size

    # Apply z-norm to both time series
    ts1 = znorm(ts1, len1)
    ts2 = znorm(ts2, len2)

    # Number of subsequences of length m in ts1 and ts2
    subs_num_ts1 = len1-m+1
    subs_num_ts2 = len2-m+1

    mp = [] # matrix profile

    # To compute the Matrix Profile, we use the Euclidean Distance to calculate the distance between subsequences.
    # To compute the ED, we need the sum of all squared values from the subsequences being compared and the dot product between the subsequences.

    # Sum of squared values for all subsequences from ts1
    sq_ts1 = []
    for i in range(0, subs_num_ts1):
        sq_ts1.append(power_sum(ts1[i:i+m], m))

    # Sum of squared values for all subsequences from ts2
    sq_ts2 = []
    for i in range(0, subs_num_ts2):
        sq_ts2.append(power_sum(ts2[i:i+m], m))

    # In a STOMP implementation, we reuse the dot products from previous subsequences.

    old_dot_products = []
    new_dot_products = []
    aux = []

    # Compute the distance between the first ts1 subsequence and every ts2 subsequence
    for i in range(0, subs_num_ts2):
        old_dot_products.append(np.dot(ts1[0:m], ts2[i:i+m]))
        aux.append(sum([sq_ts1[0], sq_ts2[i], -2*old_dot_products[i]]))
    mp.append(min(aux))

    # For the next ts1 subsequences, reuse the values from the distance profile from the previous ts1 subsequence
    for i in range(1, subs_num_ts1):
        aux = []

        new_dot_products.append(np.dot(ts1[i:i+m], ts2[0:m]))
        aux.append(sum([sq_ts1[i], sq_ts2[0], -2*new_dot_products[0]]))

        for j in range(1, subs_num_ts2):
            new_dot_products.append(old_dot_products[j-1]-ts1[i-1]*ts2[j-1]+ts1[i-1+m]*ts2[j-1+m])
            aux.append(sum([sq_ts1[i], sq_ts2[j], -2*new_dot_products[j]]))
        mp.append(min(aux))

        old_dot_products = new_dot_products
        new_dot_products = []

    return mp

def mpdist(ts1, ts2, m):
    """
        MPDist implementation using STOMP to compute the Matrix Profile.

        Parameters
        ----------
            ts1: pandas.Series
                First time series.
            ts2: pandas.Series
                Second time series.
            m: int
                Length of the subsequences.

        Output
        ------
            MPDist: float
                Distance between the two time series.
    """

    threshold = 0.05
    mp = stomp(ts1, ts2, m)
    k = int(threshold * (len(ts1) + len(ts2)))
    sorted_mp = sorted(mp)

    if len(sorted_mp) > k:
        mpdist = sorted_mp[k]
    else:
        mpdist = sorted_mp[len(sorted_mp)-1]

    return mpdist
