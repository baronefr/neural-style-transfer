# **deep-style/samples**

In this folder there are some samples used in our report.


## /

Various pictures with different settings, for display purpose. Look at the generation script for each specific setting.

| content photo | styles |
| ---           | --- |
| camogli   | starry night, sturm, turner, wave |
| prato     | starry night, sturm, tree |
| specola   | the scream, starry night, sturm, tree, turner |
| tiger     | matisse, picasso | udnie |
| tuebingen | picasso, scream, starry night, turner |


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

Starry night + Golden Gate, using each time just few conv layers for the style features.

| file               | style layer(s)   |
| ---                | ---              |
| starry-gate_conv1  | conv1_1          |
| starry-gate_conv2  | conv2_1          |
| starry-gate_conv3  | conv3_1          |
| starry-gate_conv4  | conv4_1          |
| starry-gate_conv5  | conv5_1          |
| starry-gate_conv12 | conv1_1, conv2_1 |
| starry-gate_conv15 | conv1_1, conv5_1 |
| starry-gate_conv23 | conv2_1, conv3_1 |
| starry-gate_conv45 | conv4_1, conv5_1 |

We see that the best results are obtained combining all the layers... which means, all the other pictures in the sample folder.




## weight/

We probe the `content-weight` flag.

* Sometimes the weight seems to have little effect, like in the `bolzano` series.



## cc/ 

We test the color control features: `none`, `hist`, `hist_from_style`, `luminance`. We use as reference `bolzano` and `garden`, using the style from Starry Night.



## adam/

We use Adam instead of LBFGS. The results are obviously worse.



## huang_comparison/

Plots of figure from Huang paper, made with Gatys algorithm for comparison.


## huang/

Plots of figure from Huang paper, made with Huang algorithm for comparison.
