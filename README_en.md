
# My Paper Title

This repository is the official implementation of [RCGAN-Based Inverse Design and Multi-Objective Optimization Using Customizable Weight Vectors for Lattice Structure Design ](xxx). 

**Abstract:**

    Mechanical metamaterials are engineered structures designed to exhibit unique and extraordinary mechanical properties. Traditionally, their design relied on trial-and-error methods, which are slow and limited. With the rise of machine learning, inverse design methods now provide a more efficient and systematic approach. These methods allow for a broader exploration of material properties and support the integration of multifunctionality, significantly speeding up the design process. Despite the many advantages of inverse design, lattice structures often require a trade-off between compactness and manufacturability to achieve the same target properties. Furthermore, these trade-offs must be dynamically adjusted based on different additive manufacturing conditions. To address this, we propose the RCGAN-MO architecture, which simultaneously handles the inverse design and adjustable multi-objective optimization of mechanical metamaterials. The RCGAN-MO consists of two trained neural networks: a generator and a predictor, along with a weighted multi-objective optimizer, trained on a FEM dataset. As a case study, the RCGAN-MO architecture is applied to the inverse design of the relative compressive elastic moduli  for a lattice unit cell, and the impact of different weight vector values in the multi-objective optimizer is examined through 3D printed samples. The results show that: 1) The generator achieves high accuracy in both FEM simulations and compression tests, with R² values of 99.62% and 86.99%; 2) Optimizing for compactness makes the lattice unit cell harder to print, while prioritizing manufacturability improves printability, although it leads to an increase in lattice size.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

The initial dataset used for machine learning is below:
```dataset
ML\dataset\data_for_ml.csv
```

## Parameter Configuration

Most hyperparameters are defined through the configuration file. If changes are needed, you can directly locate the relevant parameters in the file and modify them:

```parameter
ML\configs\nn.json
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
