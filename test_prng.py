from hp_prng import HpPrng

# Case 1 - test with default seed
print("\n+++++ Case 1")
g1 = HpPrng()
s = g1._get_seed()
if s == 0.0:
    print("Pass")
else:
    print("      FAILED. Stored seed is {}".format(s))

# Case 2 - test with non-zero seed
print("\n+++++ Case 2")
g2 = HpPrng(999)
s2 = g2._get_seed()
if s2 > 0.0:
    s1 = g1._get_seed()
    if s1 != s2:
        print("!     Case 2 seed ({}) different from case 1 seed ({})".format(s2, s1))
    else:
        print("Pass (seed is {})".format(s2))
else:
    print("      FAILED. Stored seed is {}".format(s))

# Case 3 - illegal seed type
print("\n+++++ Case 3")
try:
    g3 = HpPrng(24.83)
    print("      FAILED. Exception not raised.")
except TypeError as e:
    print("Pass")

# Case 4 - simple sampling of the randoms
print("\n+++++ Case 4")
seeds = [0, 17, 288, 5014, 98623, 404757]
for seed in seeds:
    counts = [0]*100
    g4 = HpPrng(seed)
    for i in range(100000):
        val = g4.random()
        index = int(val*100.0)
        counts[index] += 1

    passed = True
    for i in range(100):
        if counts[i] < 800  or  counts[i] > 1200:
            passed = False
            print("      Seed {} had count[{}] = {}".format(seed, i, counts[i]))
    if passed:
        print("      Seed {} passed.".format(seed))
    else:
        print("FAILED for seed {}".format(seed))
