##
#  This module defines the CSVScraper subclass 
#
import pandas as pd
from loaders.CSVLoader import CSVLoader

## This class allows to download CSVs from the Internet
#
class CSVScraper(CSVLoader):
    ## Constructs the CSVScraper object 
    #  @param url the url where the desired csv file is located
    #
    def __init__(self, x_names, y_names, url):
        self._url = url
        super().__init__(x_names, y_names)

    ## Overrides the superclass method and downloades a CSV using the url provided to constructor
    #
    def _load(self):
        try:
            self._df = pd.read_csv(self._url)
        except Exception as e:
            raise Exception(f'Failed to download a CSV file due to the following error: {e}')
        else:
            super()._load()
