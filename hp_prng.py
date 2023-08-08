import math

"""A pseudo-random number generator (PRNG) to replace the default one provided by Numpy.
    This algorithm was obtained from a programming manual for the HP-15C scientific calculator,
    c. 1978, and has been shown to reliably produce a uniformly random sequence of values
    (warning not tested to the level required for security applications).  Evaluations were
    made per "The Art of Computer Programming, Vol 2, Seminumerical Methods" by Donald Knuth.
"""

class HpPrng:

    _seed = 0.0     #seed value is always in [0, 1)

    def __init__(self,
                 seed : int = 0   #if provided, its value must be > 0
                ):
        """Initializes the PRNG object, possibly with a positive integer seed."""

        if isinstance(seed, int)  and  seed >= 0:
                HpPrng._seed = float(seed) / (math.pow(2, 32) - 1.0) #Python can handle larger values, but this is good
        elif seed != None:
            raise TypeError("HpPrng requires an integer seed >= 0. Given type & value was: {}, {}".format(type(seed), seed))


    def random(self):
        """Returns a pseudo-random value in [0, 1)"""

        rn = 9821.0*(self._seed + 0.211327)
        self._seed = rn - int(rn)
        return self._seed


    def _get_seed(self):
        """FOR TESTING ONLY!"""
        return self._seed
