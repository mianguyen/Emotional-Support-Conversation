"""Contains inputters"""

from inputter.esc.esc_vanilla import ESCInputter as esc_vanilla
from inputter.esc.esc_strat import ESCInputter as esc_strat



inputters = {
    'esc_vanilla': esc_vanilla,
    'esc_strat': esc_strat,
}



