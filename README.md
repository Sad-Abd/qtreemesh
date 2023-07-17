<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Made with love in SUT (Iran)][sut-badge]][sut-add]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Sad-Abd/qtreemesh">
    <img src="images/logo.png" alt="Logo" width="320" height="80">
  </a>

<h3 align="center">QTREEMESH</h3>

  <p align="center">
    Generation of QuadTree mesh from an image
    <br />
    <a href="https://github.com/Sad-Abd/qtreemesh"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Sad-Abd/qtreemesh">View Demo</a>
    ·
    <a href="https://github.com/Sad-Abd/qtreemesh/issues">Report Bug</a>
    ·
    <a href="https://github.com/Sad-Abd/qtreemesh/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

QTREEMESH is a python package that can create a [Quadtree](https://en.wikipedia.org/wiki/Quadtree) structure from an image. This tree data structure can also be converted to mesh structure that can be used in different areas of science, e.g. finite element analysis. The Quadtree algorithm in this package is based on pixels' intensity.    

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[![python][python]](https://www.python.org)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This part explains how to install and use this package.

### Installation
Install `QTREEMESH` from PyPI via pip.
```sh
pip install qtreemesh
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

There is a `test.py` file in `examples` folder that demonstrate how different parts of this package work. Here we go through this file line by line:

### 1. Read image

First we import required tools from other libraries
```python
from PIL import Image # to read image file properly
from numpy import asarray # for converting image matrix to array
```

Then we read the image and convert it to gray-scale. There are three example images in `examples` folder. `4.jpg` is smaller than the two others and need fewer computation efforts.
```python
im = Image.open("4.jpg").convert('L')
```

### 2. Preprocessing

In order to implement QuadTree algorithm, the image should be square and the number of pixels in each dimension should be of order $2^n$ (*more explanation to be added*). There is a function `image_preprocess` dedicated to this modification of original image:
```python
from qtreemesh import image_preprocess

imar = image_preprocess(asarray(im))
```

### 3. QuadTree Algorithm

The QuadTree decomposition can be performed on `image_array` using a recursive class `QTree` based on given `tolerance`.
```python
from qtreemesh import QTree

quad = QTree(None, imar, 125) # QTree(None, image_array, tolerance)
```

`QTree` object may have 4 children `QTree` objects (can be accessed through attributes: `north_west`,
`north_east`,
`south_west`,
`south_east`) and so on. Each `QTree` has an attribute `divided` that determines the existence of children partitions. There are also an property method for counting `count_leaves` and a method for saving tree leaves `save_leaves` (i.e. undivided partitions).

### 4. Mesh Generation
Common mesh data structure can be extracted from QuadTree structure using `QTreeMesh` class. After initiating the class, corresponding `elements` and `nodes` can be generated as attributes of the `QTreeMesh` object with the method `create_elements`. The resulted mesh may be illustrated using `draw` method. 
```python
from qtreemesh import QTreeMesh

mesh = QTreeMesh(quad)
mesh.create_elements()
mesh.draw(True, 'orangered') # mesh.draw(fill_inside, edge_color, save_name)
```

Each element in `elements` is a `QTreeElement` object that contains many attributes, e.g. element number : `number`, element nodes : `nodes_numbers`, element property (average of pixel intensities) : `element_property` and etc.

| Example   |      Image      |  Mesh |
|----------|:-------------:|:------:|
| 4.jpg |  <img src="examples/4.jpg" alt="image 4" width="200px"> | <img src="examples/4_meshed.png" alt="image 4 meshed" width="200px"> |
| 5.jpg |    <img src="examples/5.jpg" alt="image 5" width="200px">   |   <img src="examples/5_meshed.png" alt="image 5 meshed" width="260px"> |

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Completing the codes documentation
- [ ] Adding details to README file
- [ ] Exporting data as `vtk` format
- [ ] Successfully implement in FEM software
  - [ ] Handling hanging nodes
  - [ ] Prepare required data
  - [ ] Illustrate usage in open-source FEM programs
- [ ] Prepare required data for SBFEM


See the [open issues](https://github.com/Sad-Abd/qtreemesh/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Sadjad Abedi -  AbediSadjad@gmail.com

Project Link: [https://github.com/Sad-Abd/qtreemesh](https://github.com/Sad-Abd/qtreemesh)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Sad-Abd/qtreemesh.svg?style=for-the-badge
[contributors-url]: https://github.com/Sad-Abd/qtreemesh/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Sad-Abd/qtreemesh.svg?style=for-the-badge
[forks-url]: https://github.com/Sad-Abd/qtreemesh/network/members
[stars-shield]: https://img.shields.io/github/stars/Sad-Abd/qtreemesh.svg?style=for-the-badge
[stars-url]: https://github.com/Sad-Abd/qtreemesh/stargazers
[issues-shield]: https://img.shields.io/github/issues/Sad-Abd/qtreemesh.svg?style=for-the-badge
[issues-url]: https://github.com/Sad-Abd/qtreemesh/issues
[license-shield]: https://img.shields.io/github/license/Sad-Abd/qtreemesh.svg?style=for-the-badge
[license-url]: https://github.com/Sad-Abd/qtreemesh/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/seyed-sadjad-abedi-shahri
[product-screenshot]: images/screenshot.png
[python]: https://www.python.org/static/community_logos/python-logo.png
[sut-add]: https://sut.ac.ir
[sut-badge]: https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-SUT%20(Iran)-0c674a?style=for-the-badge
