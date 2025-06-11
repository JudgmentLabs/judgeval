"""
Error handling for scorers

"""


class MissingExampleParamsError(Exception):
    """
    Error raised when a scorer is missing required example parameters.
    """

    pass
