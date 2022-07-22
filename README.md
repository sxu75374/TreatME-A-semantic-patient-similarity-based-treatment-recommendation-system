<div id="top"></div>

# TreatME - A semantic patient-similarity-based treatment recommendation system


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#contents">Contents</a></li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#built-with">Built With</a></li>
      <ul>
          <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#author">Author</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project
<!--介绍论文--> Nowadays, medication errors happen frequently, leading to severe consequences. 42% of the medication errors are caused by the lack of medical knowledge of new drugs or experience with some diseases. In recent years, medical recommendation has played an important role in the related fields. In this paper, we constructed a semantic patient-similarity-based collaborative filtering treatment recommendation system, called TreatME. We used lexical databases **ConceptNet and NLTK WordNet** to help measure word semantic similarity and sentence semantic similarity by treating the patient’s symptoms as a sentence including semantic and structural information. Then, we implemented the collaborative filtering approach on the recommendation system to recommend top-k treatments based on the top 3 similar patients. Our proposed model shows a higher precision@k, recall@k, and f1@k score compared with the trivial model, popularity-based baseline models, and Jaccardsimilarity-based model. The performance of TreatME is promising to serve as a treatment recommendation system application in daily use based on a large dataset.

web scrapper
<!--引用算法和论文-->. 

## Contents
There are four main parts in `WholeProject.py`. 



## Screenshots
<br />
<div align="center">
  <img src="screenshots/screenshot1.png" alt="screenshot1" width="570" height="400">
</div>


## Built With
- [Python 3.7.4](https://www.python.org/downloads/release/python-374/)


### Installation
This project was built and tested with Python 3.7.4, included package nltk 3.7, scikit-learn 1.0.1, pandas 1.3.4, numpy 1.21.4, request 2.27.1 and matplotlib 3.4.3.


## further improvement

xxx


## Contribution

**Shuai Xu** | University of Southern California | [Profile](https://github.com/sxu75374) - <a href="mailto:imshuaixu@gmail.com?subject=Nice to meet you!&body=Hi Shuai!">Email</a>

Shuai Xu contributed to this project and paper individually.

### Acknowledge

This project was suppervised by Prof. Srivastava @USC and Prof. Raghavendra @USC. Thanks for their help and suggestions.

### Reference

https://conceptnet.io/

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">[<a href="#top">back to top</a>]</p>
