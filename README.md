### StackOverflow Tag Prediction

- `EDA.ipynb` includes the exploratory data analysis experiments
- `Data preprocessing.ipynb` includes the experiments for the preprocessing of the dataset
- `Feature engineering.ipynb` includes the experiments for the extraction of the features for the baseline model
- `Baseline model.ipynb` includes the experiments for the training and evaluation of the baseline model
- `LLM-based model.ipynb` includes the experiments for the training and evaluation of the LLM-based approach (not finished)

#### Run with docker

* Build the image:
    * `docker build  . -f docker/<protocol>.Dockerfile -t tag_prediction`
      
* Run the container: 
    * `docker run  -v <local/data/path>:<local/data/path> -d -p <host-port>:8080 tag_prediction`
    