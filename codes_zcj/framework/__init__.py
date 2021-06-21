"""Contains models"""

from framework.esc.esc_vanilla import ESC as esc_vanilla
from framework.esc.esc_strat import ESC as esc_strat



frameworks = {
    'esc_vanilla': esc_vanilla,
    'esc_strat': esc_strat,
}

