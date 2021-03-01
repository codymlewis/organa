# Organa
A modular HTTP-based Federated learning system

## Installation
Run
```
pip install -r requirements.txt && make && pip install .
```

## How to use
You need to construct a model that implements organa.Model, a dataset that
implements organa.DatasetWrapper, and a global model update function. These each
are used for the constructing the server and clients, the server starts on
constructions and the client must perform a look of gets and posts.

There is an example in the sample folder
