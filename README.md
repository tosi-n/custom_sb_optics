Data Capturing Pipeline
======================
Version 1.1

#### Requirements & Setup
    Python 3.7.12

With VENV Create a new virtual environment and install packages.

    virtualenv -p python3 venv

    source ./venv/bin/activate

Install requirements in virtual environment

    pip3 install -r requirements.txt


#### Repo Component Description & Dependencies

![Repo Directory and Components](misc/dir_01.png)

- Root component folders as seen in the image above contains end to end architechture and dependencies to enable training and deployment access to frontend 

![Repo Directory and Components](misc/dir_02.png)

- As seen above, the key pipeline building blocks includes data capturing architecture scripts such as:
    - [Data preprocessor script](src/custom_sb_optics/pre_process.py):
        - Contains annotator parser, data split for input full transcript list and annotated label dictionary data
            - todo: data augmentation
    - [Config arguments](src/custom_sb_optics/config/global_args.py and src/custom_sb_optics/config/utils):
    - Losses for transformer neural network loss functions
    - Ner dataset loader for torch data loading implementation via huggingface
    - [Trainer script](src/custom_sb_optics/train_runner.py):
        - Contain NER architectural transformer for training clean custom data capture data on new document type features
    - [Predictor script](src/custom_sb_optics/prediction.py): 
        - For wrapping api endpoint for trained document type data capture via workspace and trained model weights

Test pipeline in your commandline using:

    python3 ./main.py

#### What next

- Backend deployment
- Review data structure for multiple file input requiring index per file saved in nested json
- Iteration various document scenario for higher accuracy
- Benchmark result implementation for backend communication with frontend or client as messaging to notify model training completion
- Integrate NLP spell checker for postprocessing