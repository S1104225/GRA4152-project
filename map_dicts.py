import statsmodels.api as sm

from loaders.CSVScraper import CSVScraper
from loaders.CSVReader import CSVReader
from loaders.CSVStatsLoader import CSVStatsLoader

from models.GLMNormal import GLMNormal
from models.GLMBernoulli import GLMBernoulli
from models.GLMPoisson import GLMPoisson

dataset_map = {
    'duncan': {
        'name': 'duncan',
        'loader': CSVStatsLoader,
        'x_names': [ 'education', 'prestige' ],
        'y_names': [ 'income' ]
    },
    'spector': {
        'name': 'spector',
        'loader': CSVStatsLoader,
        'x_names': [ 'GPA', 'TUCE', 'PSI' ],
        'y_names': [ 'GRADE' ]
    },
    'warpbreaks': {
        'name': 'https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv',
        'loader': CSVScraper,
        'x_names': [ 'wool', 'tension' ],
        'y_names': [ 'breaks' ]
    },
}

model_map = {
    'normal': {
        'model': GLMNormal,
        'reference': sm.families.Gaussian()
    },
    'bernoulli': {
        'model': GLMBernoulli,
        'reference': sm.families.Binomial()
    },
    'poisson': {
        'model': GLMPoisson,
        'reference': sm.families.Poisson()
    },
}