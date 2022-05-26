# Implementation of L1, L2, ElasticNet, GroupLasso and GroupSparseRegularization

1. Publication available here: [https://towardsdatascience.com/different-types-of-regularization-on-neuronal-network-with-pytorch-a9d6faf4793e]
1. Implemented in __pytorch__. This is an attempt to provide different type of regularization of neuronal network weights in pytorch.
2. The regularization can be applied to one set of weight or all the weights of the model

# Metrics Scores table
| Regularization     | Test Accuracy           | Best HyperParameters  |
| ------------- |:-------------:| -----:|
| **L1**    | 98.3193 | 'batch_size': 32, 'ld_reg': 1e-05, 'lr': 0.0001, 'n_epoch': 200 |
| **L2**    | 99.1596 | 'batch_size': 32, 'ld_reg': 1e-06, 'lr': 0.0001, 'n_epoch': 200 |
| **EL**    | 98.3193 | 'alpha_reg': 0.9, 'batch_size': 32, 'ld_reg': 1e-05, 'lr': 0.001, 'n_epoch': 200 |
| **GL**    | 97.4789 | 'batch_size': 32, 'ld_reg': 1e-07, 'lr': 0.0001, 'n_epoch': 200 |
| **SGL**    | 76.4705 | 'batch_size': 128, 'ld_reg': 1e-06, 'lr': 1e-05, 'n_epoch': 200 |
| **FC**    | 90.7563 | 'batch_size': 128, 'lr': 0.01, 'n_epoch': 200 |
| **FC with Weight decay**    | 99.1596 | 'batch_size': 32, 'lr': 0.0001, 'n_epoch': 200, 'weight_decay': 0.01 |

# Sparsity Percentage table
| Model     | Layer 1 (%)         | Layer 2 (%) | Layer 3(%) |
| ------------- |:-------------:| -----:| -----:|
| **L1**    | 60 | 80 | 0 |
| **L2**    | 62.5 | 5 | 0 |
| **EL**    | 85 | 80 | 30 |
| **GL**    | 7.5 | 5 | 0 |
| **SGL**   | 92.5 | 85 | 30 |
| **FC**    | 0 | 0 | 0 |
| **FC with Weight decay** | 0 | 0 | 0 |
