import numpy as np
import statsmodels.api as sm
import argparse

# I saved mapping dictionaries in a separate file to preserve space in this code.
# If you want to see all my model names and other details, you can find them in map_dicts.py
from map_dicts import model_map, dataset_map

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''\
Custom GLMs Testing Program
----------------------------------------------------------------
This program allows to run custom-built GMLs on different datasets.
They utilize a range of concepts from OOP (e.g inheritence, polymorphism, abstract
methods, etc). It allows to slightly modify the models by selecting predictors, 
setting test_size and choosing whether to include an intercept. Finally,this program
allows to compare results of the models with the respective models from statsmodels
package.
''',
    epilog='''\
----------------------------------------------------------------
P.S. Estimations computed by respective models are not exactly the same, 
so a margin of error (1e-5) is used to compare them to check if they are approximately equal
'''
)

parser.add_argument('-d', '--dset', default='duncan', choices=[ 'duncan', 'spector', 'warpbreaks' ], help='you can choose one of the following datasets: duncan, spector, warpbreaks')
parser.add_argument('-m', '--model', default='normal', choices=[ 'normal', 'bernoulli', 'poisson' ], help='you can choose one of the following GLMs: normal, bernoulli, poisson')
parser.add_argument('-p', '--predictors', default=[ 'x1' ], nargs='+', help='specify between 1 and 3 predictor variables (e.g. -p x1 x3)')
parser.add_argument('-ts', '--test-size', type=float, default=0.3, help='set a fraction of the dataset between 0 and 1 that will be used for testing')
parser.add_argument('-rs', '--random-state', type=int, default=0, help='set a random state between 0 and 1000 for a train-test split')
parser.add_argument('-ai', '--add-intercept', action='store_true', help='specify whether to include an intercept in the model estimation')
parser.add_argument('-ps', '--print-summary', action='store_true', help='indicate whether to print model summaries or not')

args = parser.parse_args()

# Here I want to convert a list of predictors [ x1, x3 ] to indices of columns -> [ 0, 2 ] and filter incorrect values
args.predictors = [ int(i[1]) - 1 for i in args.predictors if (len(i) == 2 and i[1].isnumeric()) ]

# Here I am also filtering improper values for test_size and random_state
if (args.test_size < 0 or args.test_size > 1): raise Exception(f'--test-size should be between 0 and 1. Got: {args.test_size}')
if (args.random_state < 0 or args.random_state > 1000): raise Exception(f'--random-state should be between 0 and 1000. Got: {args.random_state}')

# Here I am retrieving all infomation about a dataset selected by user and trying to filter bad predictors (e.g. x4 since we don't have 4th column)
dataset = dataset_map[args.dset]
if (max(args.predictors) >= len(dataset['x_names']) or min(args.predictors) < 0 or len(args.predictors) == 0):
    raise Exception(f'Invalid predictors are given. Max number of predictors: {len(dataset["x_names"])}')

x_names = np.array(dataset['x_names'])[args.predictors]

# A great example of polymorphism since I do not know the exact type of CSVLoader but still I can use it
loader = dataset['loader'](x_names, dataset['y_names'], dataset['name'])
if (args.add_intercept): loader.add_constant()

x_train, x_test, y_train, y_test = loader.test_train_split(args.test_size, args.random_state)

# Another example of polymorphism since I do not know the exact type of GLModel but still I can use it
model = model_map[args.model]
glm = model['model'](x_train, y_train, args.add_intercept)
glm.fit()
if (args.print_summary): glm.summary()

# Fitting a statsmodels' GLM to compare the results
sm_glm = sm.GLM(y_train.T, x_train.T, family=model['reference'])
res = sm_glm.fit()
if (args.print_summary): print(res.summary())

glm_pred = glm.predict(x_test)
sm_glm_pred = res.predict(x_test.T)

# Setting a margin of error (MoE) to compare results since they are not exactly identical 
moe = 1e-5

# I added some nice formatting to compare the results from two modules
# Basically this part iterates over two lists at the same time and fills rows of 
# a comparison table. 
params_comparison = "\n".join([f'x{i + int(not args.add_intercept)}: {glm:>12.6f} | {sm_glm:>12.6f}' for i, (glm, sm_glm) in enumerate(zip(glm.params, res.params))])
params_summary = f'''
{"Model: " + args.model:>16} | Dset: {args.dset:<11}
-------------------------------
       My GLM    |    sm.GLM   
-------------------------------
{params_comparison}
-------------------------------
Are betas identical with MoE of {moe}: {np.allclose(glm.params, res.params, atol=moe)}
    '''

print(params_summary)


mus_comparison = "\n".join([f'{i + 1:>2}: {glm:>12.6f} | {sm_glm:>12.6f}' for i, (glm, sm_glm) in enumerate(zip(glm_pred, sm_glm_pred))])
mus_summary = f'''
{"Model: " + args.model:>16} | Dset: {args.dset:<11}
-------------------------------
       My GLM    |    sm.GLM   
-------------------------------
{mus_comparison}
-------------------------------
Are predictions identical with MoE of {moe}: {np.allclose(glm_pred, sm_glm_pred, atol=moe)}
    '''

print(mus_summary)