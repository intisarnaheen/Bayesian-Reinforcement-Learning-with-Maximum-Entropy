
## Bayesian Reinforcement Learning with Maximum Entropy Finite State Controllers.
<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />

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

## About The Project


In recent years, Reinforcement Learning (RL) has been used extensively in many real life problems. The
application field varies from resource management, medical treatment design, tutoring systems robotics to
several other fields. But in RL a problem occurs during selection of action (termed as exploration/exploitation dilemma). Bayesian Reinforcement Learning(BRL) provides an elegant solution to this dilemma. BRL
algorithms incorporate prior knowledge into the algorithm. Regrettably, BRL is computationally very demanding and several scalability problems arise. Using a Finite State Controller (FSC) this scalability issue
has been addressed. Previously boltzmann function have been used to parameterize the control parameters
of the fsc. So, we used Maximum Entropy Mellowmax Policy. Also we propose a new algorithm named as
The Monte Carlo gradient estimation algorithm with maximum Entropy mellowmax policy. The algorithm
calculates the value gradient of the control parameters and updates the parameters in the direction of the
gradient using the gradient ascent algorithm. We analyze this performance of the newly proposed algorithm with several hyper-parameters. We use two toy problem structures, chain problem and grid world,
to investigate the result of these two policies. The empirical comparison between the boltzmann and the
mellowmax policy suggests an improvement in the result.

<p align="right">(<a href="#top">back to top</a>)</p>



### Used Libraries

This project was successfully built with the following libraries. To install the libraries and use the codebase, you should properly be aware of the version conflicts of tensorflow and numpy as well. But I strongly recommend to use Anaconda for that.

* [numpy](https://python.org/)
* [Tensorflow](https://nextjs.org/)
* [Gymspaces](https://reactjs.org/)
* [Matplotlib](https://vuejs.org/)
* [Scipy](https://vuejs.org/)


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

To get started with this project, the level of code interraction needed is intermediate level at least. Please follow the simple steps below to run the repo:

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>


## Results

In this section we analyze the performance of our mellowmax policy. Here, we compare the influence of the number of memory states of the fsc. It is expected that as the number of memory states increases the performance of the finite state controller should improve.

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Number%20of%20iterations.PNG)

In this section we make a comparative analysis the Monte- Carlo gradient estimation algorithm with mellowmax policy(Algorithm 7) and the Monte Carlo gradient estimation algorithm with boltzmann policy(Algorithm 6) First the results on the chain problem has been discussed. We compare our result both on cumulative reward and cumulative discounted reward.

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_a.PNG)

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_b.PNG)

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_c.PNG)

## Contribution

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


## Contact

[@Follow me on twitter](https://twitter.com/itnsir) <br>
Email me - intisar.naheen@northsouth.edu <br>
Project Link: [Bayesian Reinforcement Learning with Maximum Entropy Finite State Controllers.](https://github.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy)

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Bastial Alt](#)
* [Dr. Heinz Koeppl](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Bio-inspired Communication Systems( BCS) Lab, TU Darmstadt](https://flexbox.malven.co/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
