"""
Legend Keeper's Grimoire: Core Logic Package
"""

# Up-import the LegendKeeper class so it can be accessed directly from 'src'
from .generator import LegendKeeper

# Define what is accessible when someone does 'from src import *'
__all__ = ["LegendKeeper"]