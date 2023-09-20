<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/zuherJahshan/vital">
    <img src="vital-icon.png" alt="Logo" width="768" height="480">
  </a>

<h3 align="center">ViTAL: Vision TrAnsformer based Low coverage pathogen classification</h3>

  <p align="center">
    ViTAL, the lineage assignment algorithm proposed here, inputs a low-coverage genome, and transforms it into embedded genome fragments using the MinHash scheme, which are then fed into a transformer-based classification neural network, that outputs the most likely lineages the input genome might belong to.
    <br />
    <!--
    <a href="https://github.com/zuherJahshan/vital"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    -->
    <a href="https://github.com/zuherJahshan/vital">View Demo</a>
    ·
    <a href="https://github.com/zuherJahshan/vital/issues">Report Bug</a>
    ·
    <a href="https://github.com/zuherJahshan/vital/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#built-with">Built With</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



### Built With
[![My Skills](https://skillicons.dev/icons?i=linux,py,git,github,tensorflow)](https://skillicons.dev)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps:

```sh
git clone https://github.com/zuherJahshan/vital.git
cd vital
conda env create -f environment.yaml
conda activate vital
cd models
python3 ./vital.py -i ../fna_examples -c 32 -k 1
```

### Prerequisites

* Ubuntu Operating System is preferable, although it can work on almost any Linux-like operating system.
* conda, please follow conda download and installation in the following link: [miniconda3 installation](https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html).
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/zuherJahshan/vital.git
   ```
2. Create the virtual environment
   ```sh
   cd vital
   conda env create -f environment.yaml
   ```
4. Download pre-trained models, this step might take a while.
   ```sh
   cd models
   wget https://zenodo.org/record/8363856/files/vital_ml_models.zip?download=1 -O data.zip
   unzip data.zip
   rm data.zip
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Simple prediction
You can perform a lineage assignment.
* Assume we are located in the ```vital/models``` directory
* Assume the genomes to be assigned lineages are all separated across different fasta files, and all found under the ```./accessions``` directory
* Assume the coverage of the genomes is 4.
```sh
python3 ./vital.py -i ./accessions/ -c 4 -o results.csv
```
After the execution of this command, the results will be inside results.csv file.

### Novel lineage phylogenetic placement
The ability to recognize novel mutations and emerging lineages is of great importance to successful genome surveillance during viral pandemics.
ViTAL can perform a preliminary step for phylogenetic placement and find the closest lineages to the newly queried genome.
To identify the predicted 5 closest lineages to the existing genome, run the following command:
```sh
python3 ./vital.py -i ./accessions/ -c 4 -k 5 -o results.csv
```


Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - Zuher Jahshan - zuher1711@gmail.com

Project Link: [https://github.com/zuherJahshan/vital](https://github.com/zuherJahshan/vital)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/zuherJahshan/vital.svg?style=for-the-badge
[contributors-url]: https://github.com/zuherJahshan/vital/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/zuherJahshan/vital.svg?style=for-the-badge
[forks-url]: https://github.com/zuherJahshan/vital/network/members
[stars-shield]: https://img.shields.io/github/stars/zuherJahshan/vital.svg?style=for-the-badge
[stars-url]: https://github.com/zuherJahshan/vital/stargazers
[issues-shield]: https://img.shields.io/github/issues/zuherJahshan/vital.svg?style=for-the-badge
[issues-url]: https://github.com/zuherJahshan/vital/issues
[license-shield]: https://img.shields.io/github/license/zuherJahshan/vital.svg?style=for-the-badge
[license-url]: https://github.com/zuherJahshan/vital/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/zuher-jahshan-7a7199196/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
