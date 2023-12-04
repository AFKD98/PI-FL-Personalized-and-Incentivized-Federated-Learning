# PI-FL-Personalized-and-Incentivized-Federated-Learning
This is the code repository for the PI-FL: Personalized and Incentivized Federated Learning paper. Currently under submission at AAAI'23. Use of this code is allowed only for academic purposes.

# How to set-up

<summary>Conda (recommended)</summary>

If you don't have Conda, you can install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install).

Once installed, create a new Conda environment. We recommend using Python 3.7, but newer versions may also work. For Mac M1/M2 systems, you must use Python 3.8 or above.

```sh
conda create -n <env_name> python=3.7
```

Activate the newly created Conda environment.

```sh
conda activate <env_name>

**Notes:**

* The quotes are required if using the Zsh shell (this is the default shell for Mac).
* There should be no spaces before or after each comma.
* The Keras backend will only work for Python 3.7; therefore, it will not work for Mac M1/M2 systems.

## Split Sample Data

You can use `generate_data.py` to generate sample data on any of the integrated datasets. This script requires the following flags:

| Flag | Description | Type |
| - | - | - |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-pp <points_per_party>` | the number of data points per party | integer |

For example to generate data for **2 parties** with **200 data points** each from the **MNIST dataset**, you could run:

```sh
python examples/generate_data.py -n 2 -d mnist -pp 200
```

Run `python examples/generate_data.py -h` for full descriptions of the different options.

By default the data is scaled down to range between 0 and 1 and reshaped such that each image is (28, 28). For more information on what preprocessing was performed, check the [Keras classifier example](/examples/keras_classifier).

## Create Configuration Files

To run IBM federated learning, you must have configuration files for the aggregator and for each party.

Before generating the configs edit the aggregator IP in line 80 in generate_configs.py 

#On line 80, replace '192.168.0.231' with aggregator machine's IP.

You can generate these config files using the `generate_configs.py` script. This script requires the following flags:

| Flag | Description | Type |
| - | - | - |
| `-f <fusion>` | which fusion algorithm to run | string |
| `-m <model>` | which framework model to use (`sklearn`, `pytorch`, `keras`, `tf`) | string |
| `-n <num_parties>` | the number of parties to split the data into | integer |
| `-d <dataset>` | which data set to use | string |
| `-p <path>` | path to load saved config data | string |
| `-tn <int>` | Number of tiers | int |
| `--party-ip <IP address>` | IP address of party's machine | IP address |


The `-n <num_parties>` and `-d <dataset>` flags should be the same same as when generating the sample data. The `-p <path>` flag will depend on the generated data from the previous step, but will typically be `-p examples/data/<dataset>/random`. The backend framework for model from the `-m <model>` flag must be installed.

This script will generate config files as follows:

```sh
# aggregator config
examples/configs/<fusion>/<model>/config_agg.yml
# party configs
examples/configs/<fusion>/<model>/config_party0.yml
examples/configs/<fusion>/<model>/config_party1.yml
...
examples/configs/<fusion>/<model>/config_party<n-1>.yml
```

For example to generate the configs for a **PyTorch model** for **2 parties** using the **iterated average fusion algorithm** from the **MNIST dataset** (generated from before), you could run:

```sh
python examples/generate_configs.py -n 100 -tn 2 -d emnist -p examples/data/emnist/balanced --fusion fedavg --model pytorch --party_ip 192.168.0.232
```

This command will generate the following config files:

```sh
# aggregator config
examples/configs/iter_avg/pytorch/config_agg.yml
# party configs
examples/configs/iter_avg/pytorch/config_party0.yml
examples/configs/iter_avg/pytorch/config_party1.yml
```

Run `python examples/generate_configs.py -h` for full descriptions of the different options.


## How does it work?

There is a [docs folder](./docs) with tutorials and API documentation to learn how to use and extend IBM federated learning. We also have a few [video tutorials](https://ibmfl.res.ibm.com/vid-intro).

- [Web Site](https://ibmfl.res.ibm.com/)
- [Aggregator and party configuration tutorial](docs/tutorials/configure_fl.md)
- [API documentation](https://ibmfl-api-docs.res.ibm.com/)

## Citing PI-FL

If you use PI-FL, please cite the following reference paper:

```bibtex
@misc{khan2023pifl,
      title={PI-FL: Personalized and Incentivized Federated Learning}, 
      author={Ahmad Faraz Khan and Xinran Wang and Qi Le and Azal Ahmad Khan and Haider Ali and Jie Ding and Ali Butt and Ali Anwar},
      year={2023},
      eprint={2304.07514},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# How to setup

Create a con
# How to Run

For generating configs use incentive_IBMfl/examples/generate_configs.py

Configs are required to generate the execution plan, python -h generate_configs.py will show the flags.

"incentive_IBMfl/examples$ python examples/generate_configs.py -n 100 -tn 2 -d emnist -p examples/data/emnist/balanced --fusion fedavg --model pytorch --party_ip 192.168.0.232"

It will generate one config file for aggregator and a config file for each party


