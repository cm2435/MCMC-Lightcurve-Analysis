"""
Exceptions for happier programming.
"""
__author__ = 'cvillforth'


class Hell(Exception):
    """
    Raise when something bad has happened.
    """
    pass


class TheDead(Exception):
    """
    Raise when something really bad has happened.
    """
    pass


class Cthulhu(Exception):
    """
    Well, can't get any worse. Only meant to be raised when the stars align again.
    """
    pass


class Hope(Exception):
    """
    Be positive, raise some hope every now and then.
    """
    pass


class OddError(Exception):
    """
    That's weird....
    """
    pass


class InputError(Exception):
    """
    Raise when the input is wrong, before doing all kinds of calculations or something absurd.
    """
    pass


class StupidError(Exception):
    """
    You made a stupid error, it's all your fault!
    """
    pass