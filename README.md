# **Semi-Supervised Generative Adversarial network for Pulsar Candidate Identification. (SGAN)**

**Procedure to Use the Code:**

1. Run the code **extract_pfd_features.py**. This reads PRESTO pfd pulsar candidates and extracts the 4 features used by the AI to classify candidates. The output of this code is a bunch of numpy array files. In order to run this code, you will need to download the following docker image https://hub.docker.com/r/sap4pulsars/pics_ai. 

2. Run the code **compute_sgan_score.py**. This code requires an anaconda3 installation along with Keras with Tensorflow backend. For quick setup, download the following docker image https://hub.docker.com/r/vishnubk/keras_gpu_docker if you plan to use GPUs for scoring (recommended). Alternatively, if you plan to only use CPUs, download the following docker image https://hub.docker.com/r/vishnubk/keras_gpu_docker.
