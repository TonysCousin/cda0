from typing import Dict
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

class CdaCallbacks (DefaultCallbacks):
    """This class provides utility callbacks that RLlib training algorithms invoke at various points."""

    _checkpoint_path = None #static class variable that will be seen by Algorithm's instance

    def __init__(self,
                 legacy_callbacks_dict: Dict[str, callable] = None
                ):
        super().__init__(legacy_callbacks_dict)
        print("///// CdaCallbacks __init__ entered. Stored path = ", CdaCallbacks._checkpoint_path)


    def set_path(self,
                 path   : str,
                ) -> None:
        """Provides an OO approach to setting the static checkpoint path location."""

        CdaCallbacks._checkpoint_path = path
        print("///// CdaCallbacks.set_path confirming that path has been stored as ", CdaCallbacks._checkpoint_path)


    def on_algorithm_init(self, *,
                          algorithm:    "Algorithm", #TODO: does this need to be "PPO"?
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

        print("///// CdaCallbacks.on_algorithm_init: checkpoint path = ", CdaCallbacks._checkpoint_path)
        # Here is an old dir from 12/17/22. It only contains one file, named checkpoint-600, so the format seems incompatible.
        ckpt = "/home/starkj/ray_results/cda0-solo/PPO_SimpleHighwayRampWrapper_53a0c_00002_2_stddev=0.6529,seed=10003_2022-12-17_10-54-12/checkpoint_000600"

        # Here is a new checkpoint made on 3/10/23 (on the Git branch tune-checkpointing, commit 4ae6)
        ckpt = "/home/starkj/projects/cda0/test/level0-pt/checkpoint_000270"
        initial_weights = algorithm.get_weights(["default_policy"])
        self._print_sample_weights("Newly created model", initial_weights)

        ### When this line is uncommented, then Ray hangs!
        temp_ppo = Algorithm.from_checkpoint(ckpt)
        print("      checkpoint loaded.")
        saved_weights = temp_ppo.get_weights()
        self._print_sample_weights("Restored from checkpoint", saved_weights)
        algorithm.set_weights(saved_weights)
        verif_weights = algorithm.get_weights(["default_policy"])
        self._print_sample_weights("Verified now in algo to be trained", verif_weights)

        """TODO: can't get the RLlib Algorithm instance of CdaCallbacks to recognize a value that I put in here.
        # if a checkpoint location has been specified, then we will attempt to load the weights from it
        if CdaCallbacks._checkpoint_path is not None:

            # Get the newly created individual NN model weights
            weights = algorithm.get_weights(["default_policy"])
            self._print_sample_weights("Newly created model", weights)

            # attempt to load the weights from the specified checkpoint file and overwrite them onto the new model
            print("///// on_algorithm_init: attempting to restore NN weights from ", CdaCallbacks._checkpoint_path)
        """


    def _print_sample_weights(self,
                              descrip   : str,
                              weights   : Dict
                             ) -> None:
        """Prints a few of the weight values to aid in confirming which model we are dealing with."""

        # Assume the NN structure is at least [10, 10] with biases and at least 20 inputs and 1 output
        print("///// Sample NN weights: {}".format(descrip))
        dp = weights["default_policy"]
        for i, dd in enumerate(dp.items()): #index 0 represents the default_policy item
            d = dd[1]
            if i == 2: #layer 0 weights (at least 10 x 20)
                print("      L0 weights: [0, 3] = {:.5f}, [1,  8] = {:.5f}, [2,  9] = {:.5f}".format(d[0, 3], d[1, 8], d[2, 9]))
                print("                  [6, 1] = {:.5f}, [7, 12] = {:.5f}, [8, 16] = {:.5f}".format(d[6, 1], d[7, 2], d[8, 6]))
            elif i == 3: #layer 0 biases (at least 10 long) (first two items are logits weights & biases)
                print("      L0 biases:  [2]    = {:.5f}, [9]     = {:.5f}".format(d[2], d[9]))
            elif i == 4: #layer 1 weights (at least 10 x 10)
                print("      L1 weights: [5, 4] = {:.5f}, [6, 6]  = {:.5f}, [7, 8]  = {:.5f}".format(d[5, 4], d[6, 6], d[7, 8]))
                print("                  [8, 0] = {:.5f}, [9, 4]  = {:.5f}, [9, 9]  = {:.5f}".format(d[8, 0], d[9, 4], d[9, 9]))
            elif i == 5: #layer 1 biases (at least 10 long)
                print("      L1 biases:  [5]    = {:.5f}, [7]     = {:.5f}".format(d[5], d[7]))
