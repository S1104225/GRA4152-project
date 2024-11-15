##
#  This module defines the CSVReader subclass 
#
import pandas as pd
from loaders.CSVLoader import CSVLoader

## This class allows to read CSV files
#
class CSVReader(CSVLoader):
    # This class variable is shared among all instances of CSVReader
    _foldername = './datasets'
    ## Constructs the CSVReader object 
    #  @param filename the filename of the dataset
    #
    def __init__(self, x_names, y_names, filename):
        self._filename = filename
        super().__init__(x_names, y_names)

    ## Overrides the superclass method and reads the CSV file using the folder and filenames
    #
    def load(self):
        try:
            self._df = pd.read_csv(f'{CSVReader._foldername}/{self._filename}')
        except Exception as e:
            raise Exception(f'Failed to read the CSV file due to the following error: {e}')
        else:
            super().load()
