# CSP_Graph

## File Structure

- Key components:

  - dataset.py : process the dataset.
  - utils.py : query related functions.
  - preprocessing.py : graph related functions.
  - models.py : model related functions.

- Main implements:
  - 2d_demo.py : demo for proposal.
  - 2d_real.py : 2d implement of test-1.csv (100 query).
  - 2d_real.ipynb : 2d implement of test-2.csv (100 query).

## Goal

We use graph method (GCN, etc.) to solve Constraint Satisfaction Problems (CSPs), start with 2D, 3D, and expand it toward up to 10 dimensions if possible.

## Fomulation

Consider a 2D version: For a bivariate distribution, represented by x, y, we have the following constraints:

### Constraints

- P( X <= 1) = 1/4
- P( Y <= 2) = 2/3
- P( X <= 2, Y <= 2) = 1/3
- ...

### Models

- 2D-Grid with each node present  P( X <= i, Y <= j) 

- Directed graph with outedge to upper and right node (Monotonicity).

- Supervised learning, each node has a value represent its probability.

  ![2d_grid](./images/2d_grid.png)

### Output

- To the labeled (with color) node, the model output is similar to its ground truth.

![2d_demo_2](./images/2d_demo_2.png)

### Evaluate

- Give perfect prediction to further distribution constraints, like:
- P( X <= 1, Y <= 3) = ?
- P( X <= 4) = ?


## Language and possible package

- Python
- PyTorch
- torch_geometric
- Networkx

