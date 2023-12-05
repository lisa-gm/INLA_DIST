# Rscripts to Preprocess and Export Data

- Synthetic datasets. contains everything that is needed to generate the synthetic datasets as described in Section 5.1 of the [paper](https://arxiv.org/abs/2303.15254). main file: ```generate_synthetic_dataset.R```
- Temperature dataset: containas dataset as described in Section 5.2. main file: ```fitting.R```.
 
In both cases you can manually choose the spatial and temporal grid sizes, as well as a number of other parameters. If ```write_to_file = TRUE```, the script will export the necessary information to file in the ```base_path``` folder (default location is ```../data```).
More about the theoretical background to our model formulation can also be found in this [paper](https://arxiv.org/abs/2006.04917).
Some additional packages are needed which you can find in the scripts itself.
