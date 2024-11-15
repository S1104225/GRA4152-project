##
#  This module defines the CSVReader subclass 
#
import pandas as pd
import statsmodels.api as sm
from loaders.CSVLoader import CSVLoader

## This class allows to read CSV files
#
class CSVStatsLoader(CSVLoader):
    ## Constructs the CSVReader object 
    #  @param filename the filename of the dataset
    #
    def __init__(self, x_names, y_names, name):
        self._name = name
        super().__init__(x_names, y_names)

    ## Overrides the superclass method and loads a sm's in-built dataset using the name provided in constructor
    #
    def load(self):
        try:
            if (self._name == 'duncan'): self._df = sm.datasets.get_rdataset('Duncan', 'carData').data
            elif (self._name == 'spector'): self._df = sm.datasets.spector.load_pandas().data
            else: raise Exception('Dataset with the given name has not been found')
        except Exception as e:
            raise Exception(f'Failed to read the CSV file due to the following error: {e}')
        else:
            super().load()
