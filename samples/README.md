# **deep-style/samples**

In this folder there are some samples used in our report.


## /



## lr/

We probe different learning rates: `1e-3`, `1e-2`, `1e-1`, `1e0`, `1e+1`, for a fixed number of iterations (600).

| style photo  | content photo | learning rate        | iterations | color control |
| ---          | ---         | ---                         | --- | ---             |
| starry night | golden gate | 1e-3, 1e-2, 1e-1, 1e0, 1e+1 | 600 | hist from style |
| "            | dinant      | 1e-3, 1e-2, 1e-1, 1e0, 1e+1 | 600 | hist            |

> We notice that increasing too much the learning rate (sample with 1e+1) leads to a black picture, as the loss becomes nan.


### lr_iter/

In this case, we also increase the number of iterations, to prove that with a smaller learning rate the algorithm is still able to reach an optimal style-matching target.

| style photo  | content photo | learning rate  | iterations | color control |
| ---          | ---         | ---     | ---  | ---             |
| starry night | golden gate | 1e-2    | 1500 | hist from style |
| "            | "           | 5e-2    | 1200 | hist from style |
| "            | "           | 1e-1    | 1200 | hist from style |


## layer/

## weight/

