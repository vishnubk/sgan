[![Website](https://img.shields.io/website?label=Paper&up_message=MNRAS&url=https%3A%2F%2Farxiv.org%2Fabs%2F2010.07457)](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stab1308/6273144)
[![Website](https://img.shields.io/website?label=license&up_message=GPL-3.0&url=https%3A%2F%2Farxiv.org%2Fabs%2F2010.07457)](https://github.com/vishnubk/sgan/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/vishnubk/sgan)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fvishnubk93)](https://twitter.com/vishnubk93)

# **Pulsar Candidate Identification Using Semi-Supervised Generative Adversarial Networks (SGAN).**


**How to score your Pulsar Candidates (PFD and/or AR files) ?**

1. Run the code **extract_features_for_classification.py**. This reads PRESTO pfd or DSPSR's ar pulsar candidates and extracts the 4 features used by the AI to classify candidates. The output of this code is a bunch of numpy array files. In order to run this code, you will need to download the following docker image https://hub.docker.com/r/sap4pulsars/pics_ai. 

2. Run the code **compute_sgan_score.py**. This code requires an anaconda3 installation along with Keras with Tensorflow2.X backend. For quick setup, download the following docker image https://hub.docker.com/repository/docker/vishnubk/sgan. 

If you would like to avoid docker, **compute_sgan_score.py** can be easily run by creating your own conda environment with python 3.6, keras tensorflow and any other packages you would need. https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


**How to Re-train SGAN with your own data?**

  Run the code **retrain_sgan.py**
  
  
