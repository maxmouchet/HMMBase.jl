# Notations

Symbol | Shape |             Description
:----- | :---- | :-----------------------------------
K      | -     | Number of states in an HMM
T      | -     | Number of observations
a      | K     | Initial state distribution
A      | KxK   | Transition matrix
B      | K     | Vector of observations distributions
α      | TxK   | Forward filter
β      | TxK   | Backward filter
γ      | TxK   | Posteriors (α * β)

**Before version 1.0:**

Symbol | Shape |             Description
:----- | :---- | :----------------------------------
π0     | K     | Initial state distribution
π      | KxK   | Transition matrix
D      | K     | Vector of observation distributions
