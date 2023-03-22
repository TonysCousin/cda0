from typing import Dict
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

class CdaCallbacks (DefaultCallbacks):
    """This class provides utility callbacks that RLlib training algorithms invoke at various points.
        For now, we need to hard-code the checkpoint file here, because there is no clear way to pass that info in.
        This needs to be the path to an algorithm-level directory. This class currently only handles a one-policy
        situation (could be used by multiple agents), with the policy named "default_policy".
    """

    # Here is a new checkpoint made on 3/10/23 (on the Git branch tune-checkpointing, commit 4ae6)
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level0-pt/05f87/checkpoint_000270"

    # More recent, collected on 3/13/23. Well-trained level 0 model (mean_reward ~10, min_reward > 0)
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level0-pt/0d8e1/PPO_00001/checkpoint_000420"

    # Level 0 completed on 3/16/23:
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level0-pt/7ea15/trial01/checkpoint_000099"

    # Level 1 completed on 3/16/23:
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level1-pt/ef817/trial13/checkpoint_000371"

    # Level 0 completed on 3/18/23:
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level0-pt/e43eb/trial00/checkpoint_000328"

    # Level 2 completed on 3/19/23:
    #_checkpoint_path = "/home/starkj/projects/cda0/training/level2/eee95/trial10/checkpoint_000171"

    # Level 3 completed on 3/22/23:
    _checkpoint_path = "/home/starkj/projects/cda0/training/level3/7c62d/trial09/checkpoint_001000"

    #_checkpoint_path = None

    #####################################################################################################

    def __init__(self,
                 legacy_callbacks_dict: Dict[str, callable] = None
                ):
        super().__init__(legacy_callbacks_dict)
        #print("///// CdaCallbacks __init__ entered. Stored path = ", CdaCallbacks._checkpoint_path)


    def set_path(self,
                 path   : str,
                ) -> None:
        """Provides an OO approach to setting the static checkpoint path location."""

        CdaCallbacks._checkpoint_path = path
        print("///// CdaCallbacks.set_path confirming that path has been stored as ", CdaCallbacks._checkpoint_path)


    def on_algorithm_init(self, *,
                          algorithm:    "PPO", #TODO: does this need to be "PPO"?
                          **kwargs,
                         ) -> None:

        """Called when a new algorithm instance had finished its setup() but before training begins.
            We will use it to load NN weights from a previous checkpoint.  No kwargs are passed in,
            so we have to resort to some tricks to retrieve the deisred checkpoint name.  The RLlib
            algorithm object creates its own object of this class, so we get info into that object
            via the class variable, _checkpoint_path.

            ASSUMES that the NN structure in the checkpoint is identical to the current run config,
            and belongs to the one and only policy, named "default_policy".
        """

        if CdaCallbacks._checkpoint_path is None:
            return
        print("///// CdaCallbacks invoked to restore model from checkpoint ", CdaCallbacks._checkpoint_path)

        # Get the initial weights from the newly created NN and sample some values
        initial_weights = algorithm.get_weights(["default_policy"])["default_policy"]
        self._print_sample_weights("Newly created model", initial_weights)

        # Load the checkpoint into a Policy object and pull the NN weights from there. Doing this avoids all the extra
        # trainig info that is stored with the full policy and the algorithm.
        temp_policy = Policy.from_checkpoint("{}/policies/default_policy".format(CdaCallbacks._checkpoint_path))
        saved_weights = temp_policy.get_weights()
        self._print_sample_weights("Restored from checkpoint", saved_weights)

        # Stuff the loaded weights into the newly created NN, and display a sample of these for comparison
        to_algo = {"default_policy": saved_weights} #should be of type Dict[PolicyId, ModelWeights]; PolicyID = str, ModelWeights = dict
        algorithm.set_weights(to_algo)    ### ERROR HERE in ndarray type conversion
        verif_weights = algorithm.get_weights(["default_policy"])
        self._print_sample_weights("Verified now in algo to be trained", verif_weights)


    def _print_sample_weights(self,
                              descrip   : str,
                              weights   : Dict
                             ) -> None:
        """Prints a few of the weight values to aid in confirming which model we are dealing with.

            ASSUMES that weights represents a single policy, not a dict of dicts.
        """

        # Assume the NN structure is at least [10, 10] with biases and at least 20 inputs and 1 output
        print("///// Sample NN weights: {}".format(descrip))

        # If the weights were loaded from an Algorithm checkpoint then the dict will be nested in a dict of policies, so first
        # need to pull out the correct policy. If the weights come from a Policy checkpoint then the wrapping dict will be
        # absent.
        dp = weights
        try:
            dp = weights["default_policy"]
        except KeyError as e:
            pass

        for i, key in enumerate(dp):
            d = dp[key]
            if i == 2: #layer 0 weights (at least 10 x 20)
                print("      L0 weights: [0, 3] = {:8.5f}, [1,  8] = {:8.5f}, [2,  9] = {:8.5f}".format(d[0, 3], d[1, 8], d[2, 9]))
                print("                  [6, 1] = {:8.5f}, [7, 12] = {:8.5f}, [8, 16] = {:8.5f}".format(d[6, 1], d[7, 2], d[8, 6]))
            elif i == 3: #layer 0 biases (at least 10 long) (first two items are logits weights & biases)
                print("      L0 biases:  [2]    = {:8.5f}, [9]     = {:8.5f}".format(d[2], d[9]))
            elif i == 4: #layer 1 weights (at least 10 x 10)
                print("      L1 weights: [5, 4] = {:8.5f}, [6, 6]  = {:8.5f}, [7, 8]  = {:8.5f}".format(d[5, 4], d[6, 6], d[7, 8]))
                print("                  [8, 0] = {:8.5f}, [9, 4]  = {:8.5f}, [9, 9]  = {:8.5f}".format(d[8, 0], d[9, 4], d[9, 9]))
            elif i == 5: #layer 1 biases (at least 10 long)
                print("      L1 biases:  [5]    = {:8.5f}, [7]     = {:8.5f}".format(d[5], d[7]))
