from typing import Dict
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from perturbation_control import PerturbationController


class CdaCallbacks (DefaultCallbacks):
    """This class provides utility callbacks that RLlib training algorithms invoke at various points.
        This needs to be the path to an algorithm-level directory. This class currently only handles a one-policy
        situation (could be used by multiple agents), with the policy named "default_policy".
    """

    def __init__(self,
                 legacy_callbacks_dict: Dict[str, callable] = None
                ):
        super().__init__(legacy_callbacks_dict)

        self.info = PerturbationController()
        self._checkpoint_path = self.info.get_checkpoint_path()
        #print("///// CdaCallback instantiated! algo counter = ", self.info.get_algo_init_count())


    def on_algorithm_init(self, *,
                          algorithm, #: "PPO", #TODO: does this need to be "PPO"?
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

        #TODO: skip for now, since this default_policy logic won't work for SAC
        return

        # Update the initialize counter
        self.info.increment_algo_init()

        # If there is no checkpoint specified, then return now
        if self._checkpoint_path is None:
            print("///// CdaCallbacks detects checpoint path is None, so returning.")
            return

        # Once perturbations begin we don't want to be loading checkpoints any more, so return
        if self.info.has_perturb_begun():
            return
        print("///// CdaCallbacks restoring model from checkpoint ", self._checkpoint_path)

        # Get the initial weights from the newly created NN and sample some values
        initial_weights = algorithm.get_weights(["default_policy"])["default_policy"]
        self._print_sample_weights("Newly created model", initial_weights)

        # Load the checkpoint into a Policy object and pull the NN weights from there. Doing this avoids all the extra
        # trainig info that is stored with the full policy and the algorithm.
        temp_policy = Policy.from_checkpoint("{}/policies/default_policy".format(self._checkpoint_path))
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
