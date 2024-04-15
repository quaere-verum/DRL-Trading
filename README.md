# Goal and modelling assumptions
This repository contains gymnasium environments and several utilities for training a deep reinforcement learning agent to become a trading bot. 
This process relies entirely on numerical features of the assets under consideration, and so this is a purely quantitative strategy.
An import assumption was made when starting this project:

> One cannot delegate the task of both classifying trading opportunities, and optimising a trading strategy, to the same model

This is reflected in the design of the environments. Indeed, it is necessary for the dataframe that is passed in as the "data" argument to contain,
for each asset under consideration, a column called "pred" which contains the predicted behaviour of the corresponding asset. For instance, one might 
fit a linear regression model, or something more complex, and use the output as the "pred" column. When opening a position, the expected return is
stored, and used as a take profit threshold. Additionally, if the position has remained open throughout the entire lookahead period over which the
return was predicted, the position will be automatically closed at the current closing price.

# How to use
1. Select your assets and obtain their pricing data.
2. Add technical indicators and design dynamic feature functions (i.e. depending on the current open positions/capital etc.)
3. Set the columns of the dataframe to be a multi index [(asset1, feature1), (asset1, feature2), ..., (assetn, featurem)]
4. Define a dictionary called "feature_properties". It should contain an entry for each technical feature you've added, and its key should be the second level of the column name (i.e. feature1 above). Its value should be a dictionary with the following keys
   - 'sequential'. Value a boolean which determines whether the observation will return historical info of this feature, or only the most recent value
   - 'gauge'. Value a boolean which determines whether the observation needs to be divided by the current closing price
5. Define a dictionary called "dynamic_features". Its key are the names of the dynamic features you've designed, and its values are dictionaries with the following keys:
   - 'sequential', see above
   - 'gauge', see above
   - 'initial_value', self-explanatory
   - 'function', a reference to the function that you've designed

Now the environment can be created. In order to train a reinforcement learning model, it is import to know the structure of the observations returned by the environment.
Each observation is a dictionary which contains three keys:
   - 'prices', which contains the price series of the assets
   - 'sequential', which contains all those features (dynamic and otherwise) which were passed with 'sequential': True
   - 'other', which contains the remaining features

In each case, they are divided (or "gauged", in physics lingo) by the most recent closing prices of the assets.
What remains is to design or choose an existing DRL algorithm, and create a feature extractor for the observations. An example of such a feature extractor is provided in the [torch modules file](https://github.com/quaere-verum/DRL-Trading/blob/main/feature_extraction/torch_modules.py).
See also main.py.
