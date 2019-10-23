import numpy as np
from sktime.utils.matrix_profile import stomp_self


def motifs(ts, m):
    """
        Finds the closest motif pair in the time series.

        Parameters
        ----------
            ts: numpy.array
                Time series.
            m: int
               Length of the subsequences.

        Output
        ------
            motif: tuple
                   Tuple with the indexes of the closest motif pair.
    """

    ts_len = len(ts)

    mp, ip = stomp_self(ts, m)

    motif = (np.argmin(mp), int(ip[np.argmin(mp)]))

    return motif
