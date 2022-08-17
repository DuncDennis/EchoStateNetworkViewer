"""Utility functions."""

import contextlib #  for temp_seed
import inspect

import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    """
    from https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    Use like:
    with temp_seed(5):
        <do_smth_that_uses_np.random>
    """

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class _SynonymDict:
    """ Custom dictionary wrapper to match synonyms with integer flags.

    CREDIT: Taken from https://github.com/GLSRC/rescomp.

    Internally the corresponding integer flags are used, but they are very much
    not descriptive so with this class one can define (str) synonyms for these
    flags, similar to how matplotlib does it

    Idea:
        self._synonym_dict = {flag1 : list of synonyms of flag1,
                              flag2 : list of synonyms of flag2,
                              ...}
    """

    def __init__(self):
        self._synonym_dict = {}
        # super().__init__()

    def add_synonyms(self, flag, synonyms):
        """ Assigns one or more synonyms to the corresponding flag

        Args:
            flag (int): flag to pair with the synonym(s)
            synonyms (): Synonym or iterable of synonyms. Technically any type
                is possible for a synonym but strings are highly recommended

        """

        # self.logger.debug("Add synonym(s) %s to flag %d"%(str(synonyms), flag))

        # Convert the synonym(s) to a list of synonyms
        if type(synonyms) is str:
            synonym_list = [synonyms]
        else:
            try:
                synonym_list = list(iter(synonyms))
            except TypeError:
                synonym_list = [synonyms]

        # make sure that the synonyms are not already paired to different flags
        for synonym in synonym_list:
            found_flag = self._find_flag(synonym)
            if flag == found_flag:
                # self.logger.info("Synonym %s was already paired to flag"
                #                  "%d" % (str(synonym), flag))
                synonym_list.remove(synonym)
            elif found_flag is not None:
                raise Exception("Tried to add Synonym %s to flag %d but"
                                " it was already paired to flag %d" %
                                (str(synonym), flag, found_flag))

        # add the synonyms
        if flag not in self._synonym_dict:
            self._synonym_dict[flag] = []
        self._synonym_dict[flag].extend(synonym_list)

    def _find_flag(self, synonym):
        """ Finds the corresponding flag to a given synonym.

        A flag is always also a synonym for itself

        Args:
            synonym (): Thing to find the synonym for

        Returns:
            flag (int_or_None): int if found, None if not

        """

        # self.logger.debug("Find flag for synonym %s"%str(synonym))

        flag = None
        if synonym in self._synonym_dict:
            flag = synonym
        else:
            for item in self._synonym_dict.items():
                if synonym in item[1]:
                    flag = item[0]

        return flag

    def get_flag(self, synonym):
        """ Finds the corresponding flag to a given synonym. Raises exception if
            not found

        see :func:`~SynonymDict._find_flag_from_synonym`

        """
        flag = self._find_flag(synonym)
        if flag is None:
            raise Exception("Flag corresponding to synonym %s not found" %
                            str(synonym))

        return flag


def _remove_invalid_args(func, args_dict):
    """Return dictionary of valid args and kwargs with invalid ones removed

    CREDIT: Taken from https://github.com/GLSRC/rescomp.

    Adjusted from:
    https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives

    Args:
        func (fct): function to check if the arguments are valid or not
        args_dict (dict): dictionary of arguments

    Returns:
        dict: dictionary of valid arguments

    """
    valid_args = inspect.signature(func).parameters
    # valid_args = func.func_code.co_varnames[:func.func_code.co_argcount]
    return dict((key, value) for key, value in args_dict.items() if key in valid_args)


def sigmoid(x):
    """The sigmoid activation function. """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """The relu activation function."""
    return x * (x > 0)
