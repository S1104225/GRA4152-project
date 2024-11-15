# GRA4152 Final Project

This repository contains two main superclasses: CSVLoader & GLMBase.

### Test File

It is possible to test the program by using a test file. See details by typing following code in the console:

```python3 testing_glm.py -h```

### CSVLoader superclass

- There are 3 subclasses of this class that utilize the notion of polymorphism: CSVReader (to read from disk), CSVScraper (to read from the Internet), CSVStatsLoader (to use in-built datasets from statsmodels package)
- One can initialise them in the following way:

**CSVLoader([names of x variables], [names of y variables], 'name of the dataset')**

- After that is loads the dataset automatically with the following shape: [p, N]
- Afterwards it is possible to access x, y, x_transpose, y_transpose
- There are also 2 methods:
    * .add_constant() - add a row of 1s to the dataset
    * .test_train_split(test_size, random_state) - splits data into training and testing parts

### GLMBase superclass

- There are 3 subclass built on the notion of polymorphism: GLMNormal, GLMBernoulli, GLMPoisson.
- They are constructed like that:

**CSVLoader([x variables], [y variables], boolean identifying if an intercept is used)**

- It is recommended to .fit() it before accessing other methods
- After that parameters can be accessed as a property .params
- There are also 2 methods:
    * .summary() - provides a summary of a model
    * .predict([new x values]) - predict new Ys based on estimated parameters