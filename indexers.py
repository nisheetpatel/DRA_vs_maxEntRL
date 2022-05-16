from typing import Any, Callable
import numpy as np
from tasks import option_choice_set_2afc, option_choice_set_bottleneck


Indexer = Callable[[list, int, int], Any]


def standard_indexer(state: list, action: int = None, n_actions: int = None) -> tuple:
    """Returns standard q-table index as tuple"""
    if (action is None) & (n_actions is None):
        return tuple(state)
    elif action is not None:
        return tuple(state + [action])
    else:
        return tuple(state + [np.s_[:n_actions]])


def memory_2afc_task_indexer(state: list, action: int = None, n_actions: int = None):
    """Returns q-table index for Memory_2AFC task as tuple"""
    choice_set = option_choice_set_2afc(state=state[0])
    if (action is None) & (n_actions is None):
        return state[0]
    elif action is None:
        return choice_set
    else:
        return choice_set[action]


def bottleneck_task_indexer(state: list, action: int = None, n_actions=None):
    """Returns q-table index for Bottleneck task as tuple"""
    choice_set = option_choice_set_bottleneck(state=state[0])
    if (action is None) & (n_actions is None):
        return state[0]
    elif action is None:
        return choice_set
    else:
        return choice_set[action]
