"""Unit tester for PerturbationController"""

from perturbation_control import PerturbationController

# Case 1 - initialize from main (params specified)
try:
    pc0 = PerturbationController("dummy_checkpoint.pt", 5)
    print("Case 1 success.")
except Exception as e:
    print("Case 1 caught exception: ", e)

# Case 2 - initiallize worker copy
try:
    pc1 = PerturbationController()
    print("Case 2 success.")
except Exception as e:
    print("Case 2 caught exception: ", e)

# Case 3 - compare checkpoint paths
try:
    path1 = pc1.get_checkpoint_path()
    path0 = pc0.get_checkpoint_path()
    if path0 == path1:
        print("Case 3 success.")
    else:
        print("Case 3 mismatch: path0 = {}, path1 = {}".format(path0, path1))
except Exception as e:
    print("Case 3 caught exception: ", e)

# Case 4 - compare counters before incrementing
try:
    c1 = pc1.get_algo_init_count()
    c0 = pc0.get_algo_init_count()
    if c0 == c1  and  c0 == 2:
        print("Case 4 success.")
    else:
        print("Case 4 incorrect counters: c0 = {}, c1 = {}".format(c0, c1))
except Exception as e:
    print("Case 4 caught exception: ", e)

# Case 5 - increment main counter and verify its value in both places
try:
    pc0.increment_algo_init()
    c0 = pc0.get_algo_init_count()
    c1 = pc1.get_algo_init_count()
    if c0 == c1  and  c0 == 3:
        print("Case 5 success.")
    else:
        print("Case 5 incorrect counters: c0 = {}, c1 = {}".format(c0, c1))
except Exception as e:
    print("Case 5 caught exception: ", e)

# Case 6 - increment worker counter and verify its value in both places
try:
    pc1.increment_algo_init()
    c0 = pc0.get_algo_init_count()
    c1 = pc1.get_algo_init_count()
    if c0 == c1  and  c0 == 4:
        print("Case 6 success.")
    else:
        print("Case 6 incorrect counters: c0 = {}, c1 = {}".format(c0, c1))
except Exception as e:
    print("Case 6 caught exception: ", e)

# Case 7 - create a new worker object
try:
    pc2 = PerturbationController()
    print("Case 7 success.")
except Exception as e:
    print("Case 7 caught exception: ", e)

# Case 8 - compare checkpoint paths
try:
    path2 = pc2.get_checkpoint_path()
    path0 = pc0.get_checkpoint_path()
    if path0 == path2:
        print("Case 8 success.")
    else:
        print("Case 8 mismatch: path0 = {}, path2 = {}".format(path0, path2))
except Exception as e:
    print("Case 8 caught exception: ", e)

# Case 9 - create a new main object with no checkpoint file
try:
    pc_new_main = PerturbationController(None, 6)
    pc4 = PerturbationController() #new worker
    path = pc4.get_checkpoint_path()
    if path is not None:
        print("Case 9 - worker path is {}. Should be None".format(path))
    print("Case 9 success.")
except Exception as e:
    print("Case 9 caught exception: ", e)
