from util import preprocess_physionet, make_data_physionet, make_knowledge_physionet, evaluate
from mina import run

if __name__ == "__main__":
    """
    before running preprocess_physionet, 
    please first download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in data_path
    """

    # prepare data
    data_path = '../../data/challenge2017/'
    # preprocess_physionet(data_path) # uncomment if you don't have the raw data
    # make_data_physionet(data_path) # uncomment if you don't have the preprocessed data
    # make_knowledge_physionet(data_path) # uncomment if you don't have the preprocessed knowledge
    
    # run
    for i_run in range(1):
        run(data_path)