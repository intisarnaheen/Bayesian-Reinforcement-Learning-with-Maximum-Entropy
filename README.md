
## Bayesian Reinforcement Learning with Maximum Entropy Finite State Controllers.
<div id="top"></div>
<p align="left"> 
<img src="https://komarev.com/ghpvc/?username=intisarnaheen&color=blueviolet" alt="watching_count" />
  <img src="https://img.shields.io/badge/Focus-Computer%20Vision%2C%20Machine%20Learning-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
</p>


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
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contribution">Open For Contribution</a></li>
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

* [numpy](https://numpy.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Gym](https://gym.openai.com/docs/)
* [Matplotlib](https://matplotlib.org/)
* [Scipy](https://scipy.org/)


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

To get started with this project, the level of code interraction needed is intermediate level at least. Please follow the simple steps below to run the repo:


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy.git
   ```
2. Install numpy
   ```sh
   pip install numpy
   ```
3. Install Tensorflow
   ```sh
   pip install tensorflow
   ```
4. Install Gym
   ```sh
   pip install gym
   ```
5. Install Matplotlib
   ```sh
   pip install matplotlib
   ```
5. Install Scipy
   ```sh
   pip install scipy
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


## Results

In this section we analyze the performance of our mellowmax policy. Here, we compare the influence of the number of memory states of the fsc. It is expected that as the number of memory states increases the performance of the finite state controller should improve.

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Number%20of%20iterations.PNG)

In this section we make a comparative analysis the Monte- Carlo gradient estimation algorithm with mellowmax policy(Algorithm 7) and the Monte Carlo gradient estimation algorithm with boltzmann policy(Algorithm 6) First the results on the chain problem has been discussed. We compare our result both on cumulative reward and cumulative discounted reward.

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_a.PNG)

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_b.PNG)

![Results_image](https://raw.githubusercontent.com/intisarnaheen/Bayesian-Reinforcement-Learning-with-Maximum-Entropy/master/Snapshots/Cumulative%20reward%20_c.PNG)

## Do you want to contribute?

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (` git checkout -b feature/Bayesian-Reinforcement-Learning-with-Maximum-Entropy `)
3. Commit your Changes (` git commit -m 'Add some Bayesian-Reinforcement-Learning-with-Maximum-Entropy' `)
4. Push to the Branch (` git push origin feature/Bayesian-Reinforcement-Learning-with-Maximum-Entropy `)
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
