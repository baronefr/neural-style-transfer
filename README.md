<h1 align="center"> <b>Neural Style Transfer</b> </h1>

<p align="center"><b>Project</b> // Barone Francesco, Ninni Daniele, Zinesi Paolo<br></p>

<p align='center'>
  <img src='samples/prato-afremov.jpg' height="140px">
  <img src='samples/tiger-picasso.jpg' height="140px">
  <img src='samples/camogli-wave.jpg' height="140px">
  <img src='samples/cc/bolzano-starry_hist.jpg' height="140px">
</p>


## Inside this repo

* `gatys.py` - This script is the final version of our implementation of *Gatys et al* (2016). It is meant to be executed by command-line only.
  - `gatys.ipynb` - Includes all the functionalities of the .py script, but embedded in a readable notebook to show the use of our custom library (deepstyle) which collects all the functions required by this NST algorithm.
  - `gatys_nolib.ipynb` - The notebook provides an overview of Gatys' algorithm without making use of our library (deepstyle).

* `huang_test.ipynb` - The model presented in *Huang et al* (2017), implemented inside a notebook. This model makes use of the .pth files in `models/`.
  - `huang_train.ipynb` - This notebook is used to train the decoder network.

<br>

## Setup

We recommend the use of PyTorch (>= 1.14.0). To install the required dependencies, please run
```bash
pip install -r requirements.txt
```



### How to **run Gatys NST**

The Gatys model is provided with a .py script:
```bash
python3 gatys.py -output example.jpg -iterations 600 \
    -content-weight 1e1 \
    -style image/starry-night.jpg \
    -content image/bolzano.jpg \
    -color-control hist
```

Some options include:
* `output` - the output style image to be written
* `style` and `content` - the images to be used as style and content features, respectively
* `color-control` - choose a color control option among `none`, `luminance`, `hist`, `hist_from_style` (default: none)
* `content-weight` - set the content weight (float) (default: 1e0)
* `iterations` - set the number of iterations to execute (default: 400)
* `image-size` - set the short-edge image resolution to be generated (default: 512)
* `high-res` - set any value greater than `image-size` to enable the upscaling to higher resolution (default: none)

### How to **run Huang NST**

The Huang algorithm is presented as Jupyter notebooks: `huang_test.ipynb` and `huang_train.ipynb`. Look inside the notebooks for further instructions.


<br><br>

## Bibliography

* L. A. Gatys, A. S. Ecker, and M. Bethge, **"Image style transfer using convolutional neural networks"**, in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2414–2423, 2016
* L. A. Gatys, M. Bethge, A. Hertzmann, and E. Shechtman, **"Preserving color in neural artistic style transfer"**, arXiv preprint arXiv:1606.05897, 2016
* X. Huang and S. Belongie, **"Arbitrary style transfer in real-time with adaptive instance normalization"**, in ICCV, 2017


---

<h5 align="center">Neural Networks and Deep Learning<br>AY 2022/2023 - University of Padua</h5>

<p align="center">
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_unipd.png" alt="" height="70"/>
  &emsp;
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_pod.png" alt="" height="70"/>
</p>
