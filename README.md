# ClearML Demo
A small demo of ClearML modelling the titanic dataset using scikit-learn and Keras.
## Setup

**pip**
```bash
pip install -r requirements.txt
```

**poetry**
```bash
poetry install
poetry shell
```

## Usage

Run the following command:
```
python main.py
```

To configure run use the following options:
```
Options:
  --project-name TEXT             Name of the project  [default: titanic-demo]
  --train-size FLOAT              Size of the training set  [default: 0.9]
  --epochs INTEGER                Number of epochs  [default: 100]
  --batch-size INTEGER            Batch size  [default: 32]
  --n-layer INTEGER               Number of layers  [default: 2]
  --n-units INTEGER               Number of units  [default: 32]
  ```