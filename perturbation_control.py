import time
import numpy as np

PBTC_STORAGE_DIR = "/tmp"

class PerturbationController:
    """Emulates a singleton pattern by storing and retrieving the same info for all clients, but accommodates clients in
        multiple threads/processes that can actually generate new instances of this class, so it is not a true singleton.
        It ensures that all instances are using the same data by storing that data in a single file system location. This
        will work for any number of processes sharing the same computing node, however, it of course comes with a speed
        penalty due to the disk accesses and sometimes lockouts while waiting on other processes to compete their access.

        This class is to be used in support of Ray's Population Based Training (PBT) scheduler, which recreates Algorithm
        objects for each trial at every perturb cycle, and therefore new environments, thus erasing all memory of past
        progress that is reachable by the environment code.  PBT will create a new instance of this class for each trial.
        Once all trials reach the perturb trigger (e.g. number of iterations complete), PBT will store temp checkpoints,
        perturb some HPs, then generate all new Algorithms and Environments (which will generate new instances of this
        class). At that point, these new instances need to wake up, find their relevant persistent info stored in the
        file system, and pick up from there. Therefore, after num_trials instantiations, we assume that we are
        now in the 2nd or later perturb cycle of a previously created trial. As such, we do NOT want to serve up
        the checkpoint path again, which would simply cause PBT to start training from the beginning all over again.

        This is the only known solution, at this time, for communicating certain custom information across Ray workers
        and Algorithm instances in a hyperparameter tuning situation with mutiple trials underway.
    """

    STORAGE_BASE_NAME   = ".pbt_env_info"
    STORAGE_EXT1        = ".1"
    STORAGE_EXT2        = ".2"

    def __init__(self,
                 checkpoint:    str = None, #Name of the checkpoint dir to be used as starting baseline
                 num_trials:    int = 0,    #Num HP search trials being executed
                ) -> None:
        """If this object is instantiated by the main program, then provide all parameters. All other callers should
            NOT provide any params.
        """

        # Initialize everything to null to indicate that nothing has yet been defined or stored.
        self._info_path         = None  #Full path to the file used to store info for this run (persistent storage for this class)
        self._checkpoint_path   = None  #Full path to the checkpoint directory to be used as the beginning baseline
        self._num_trials        = 0     #Num PBT trials being run
        self._algo_init_count   = 0     #Num initializations of the Algorithm object across the HP run
        self._info_valid        = False #Is the internal storage info valid and usable?

        # Local storage contents are in two separate text files. The first file is only written once per program execution
        # and contains the following (one item per line):
        #   checkpoint_path
        #   num_trials
        # The second file simply contains the algo_init_count.
        self._filename1 = "{}/{}{}".format(PBTC_STORAGE_DIR, PerturbationController.STORAGE_BASE_NAME, PerturbationController.STORAGE_EXT1)
        self._filename2 = "{}/{}{}".format(PBTC_STORAGE_DIR, PerturbationController.STORAGE_BASE_NAME, PerturbationController.STORAGE_EXT2)

        # The local storage files may already exist, left over from a previous run, so plan to overwrite them at the beginning of each run

        # If the input params are defined (we are the main instance that needs to initiate things) then
        if num_trials > 0: #checkpoint = None is a legitimate argument

            # Create the storage files and obtain a write lock on them
            success = False

            # Write our info to the files and close them as quickly as possible
            try:
                with open(self._filename1, "w") as f1:
                    if checkpoint is None:
                        f1.write("\n")
                    else:
                        f1.write("{}\n".format(checkpoint))
                    f1.write("{}\n".format(num_trials))
                success = True
            except:
                print("///// PerturbationController.__init__ failed to write {}".format(self._filename1))

            if success:
                self._checkpoint_path = checkpoint
                self._num_trials = num_trials

                success = False
                self._algo_init_count = 1 #this is the first instantiation in the whole program
                try:
                    with open(self._filename2, "w") as f2:
                        f2.write("{}\n".format(self._algo_init_count))
                    success = True
                except:
                    print("///// PerturbationController.__init__ failed to write {}".format(self._filename1))

            # Indicate if the local info is valid
            self._info_valid = success
            print("///// PerturbationController master instance created.")

        else: # Else (this instance is owned by an environment object in a worker thread)

            # Attempt to get the persistent data from the main instance. If not available yet, then not fatal.
            self._have_valid_data()
            #print("///// PerturbationController secondary instance created.")


    def get_checkpoint_path(self) -> str:
        """Returns the path to the starting checkpoint or None if none is to be used.
            Throws AssertionError if checkpoint has not been determined.
        """

        self._have_valid_data()
        assert self._info_valid, "///// PerturbationController.get_checkpoint_path: controller is not properly initialized."
        return self._checkpoint_path


    def get_num_trials(self) -> int:
        """Returns the number of trials in use.
            Throws AssertionError if data has not been initialized correctly.
        """

        self._have_valid_data()
        assert self._info_valid, "///// PerturbationController.get_num_trials: controller is not properly initialized."
        return self._num_trials


    def increment_algo_init(self) -> int:
        """Tallies another instantiation of an Algorithm object, and returns the new count.
            Throws AssertionError if the counter has not been properly initialized.
        """

        assert self._info_valid, "///// PerturbationController.increment_algo_init: controller is not properly initialized."

        # Open with r/w access to keep a lock on it for the whole operation, then read the current value from the file in
        # case another instance has changed it. Then increment the value and write it back out.
        success = False
        num_attempts = 0
        prng = np.random.default_rng()
        while num_attempts < 10:
            try:
                with open(self._filename2, "r+") as f2:
                    count = int(f2.readline())
                    if count >= 0:
                        f2.seek(0)
                        f2.write("{}\n".format(self._algo_init_count))
                        self._algo_init_count = count + 1
                        success = True
                        break
            except:
                print("///// PerturbationController.increment_algo_init: i/o error on {}".format(self._filename2))

            time.sleep(prng.random()*0.020 + 0.001) #wait up to 20 ms before attempting again
            num_attempts += 1

        if not success:
            raise IOError("///// PerturbationController.increment_algo_init was unable to update counter in {}".format(self._filename2))
        return self._algo_init_count


    def get_algo_init_count(self) -> int:
        """Returns the accumulated number of Algorithm objects that have been created.
            Throws AssertionError if the counter hasn't been properly initialized.
        """

        self._have_valid_data()
        assert self._info_valid, "///// PerturbationController.get_algo_init_count: controller is not properly initialized."

        # Read the latest value from the storage file
        success = False
        num_attempts = 0
        prng = np.random.default_rng()
        while num_attempts < 10:
            val = -1
            try:
                with open(self._filename2, "r") as f2:
                    val = int(f2.readline())
            except:
                print("///// PerturbationController.get_algo_init_count: read error on {}".format(self._filename2))

            if val >= 0:
                self._algo_init_count = val
                success = True
                break
            else:
                time.sleep(prng.random()*0.020 + 0.001) #wait up to 20 ms before attempting again
                num_attempts += 1

        if not success:
            raise IOError("///// PerturbationController.get_algo_init_count was unable to read counter in {}".format(self._filename2))
        return self._algo_init_count


    def has_perturb_begun(self):
        """Returns True if the first perturb cycle is either underway or complete (maybe additional cycles have completed also).
            Returns False if training is still using the initial set of HPs.
        """

        #TODO: this relationship is fragile - need to investigate how it may change as numbers of
        #       rollout workers, eval workers, and other HPs change.
        # Determine when the first perturbation cycle has been completed.
        #   On a single node running 1 env per worker and 2 evaluation workers (typically runs 2 workers simultaneously)
        #   Ray spins up approx 16 env copies per trial per perturb cycle (it usually runs a few extra for the whole cycle).
        #   To be safe, we need to allow for these extra copies when defining the threshold; allow up to 14 extras.
        max_instances = 16*self._num_trials + 14
        return self.get_algo_init_count() > max_instances


    def get_num_perturb_cycles(self) -> int:
        """Returns the number of times perturbation has been applied during the tuning run. This result is approximate, as
            the number of environment instances isn't necessarily constant for each perturb cycle (unknown why).
        """

        return int(max(self.get_algo_init_count() - 8, 0)) // 16


    def _have_valid_data(self) -> bool:
        """Looks for data files containing the internal storage data, and attempts to read them. Marks the
            member variable _info_valid appropriately and returns its value.
        """

        # If data has already been validated, then we're done
        if self._info_valid:
            return True

        # Attempt to open the storage files and read their contents, updating local data accordingly
        success = False
        num_attempts = 0
        prng = np.random.default_rng()
        while num_attempts < 20:
            s = ""
            nt = -1
            try:
                with open(self._filename1, "r") as f1:
                    s = f1.readline()
                    nt = int(f1.readline())
                    if len(s) > 0: #should at least contain a \n
                        self._checkpoint_path = s.split("\n")[0]
                        if len(self._checkpoint_path) == 0:
                            self._checkpoint_path = None
                        if nt >= 0:
                            self._num_trials = nt
                            success = True
                            break
            except:
                pass
                #print("///// PerturbationController._have_valid_data read error on {}".format(self._filename1))

            time.sleep(prng.random()*0.020 + 0.001) #wait up to 20 ms before attempting again
            num_attempts += 1

        # For the algo counter, the file should already exist, so increment the counter normally.
        if success:
            try:
                self._info_valid = True #set this first, as increment_algo_init() checks it!
                self.increment_algo_init()
            except Exception as e:
                print("///// PerturbationController._have_valid_data exception trapped: ", e)

        return self._info_valid
