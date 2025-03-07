# Modeling Electricity Consumption Patterns in Buildings Using Probability Distribution Functions

This project aims to model electricity consumption patterns in buildings using various probability distribution functions. The project includes data processing, distribution fitting, and result visualization.

## Project Structure

```
Modeling-Electricity-Consumption-Patterns-in-Buildings-Using-Probability-Distribution-Functions/
│
├── datasets/                   # Directory containing the datasets
│   └── [dataset_name]/         # Subdirectories for each dataset
│       └── [dataset_file].csv  # CSV files with electricity consumption data
│
├── results/                    # Directory to store the results
│   └── [dataset_name]/         # Subdirectories for each dataset's results
│       └── [result_files].csv  # CSV files with the results of distribution fitting
│
├── src/                        # Source code directory
│   ├── utils.py                # Utility functions
│   ├── mixture_models_kde.py   # Custom mixture models and KDE implementations
│   ├── fitOneDistDataCategory.py # Script to fit a single distribution to data
│   ├── fitDistributionsDataCategory.py # Script to fit multiple distributions to data
│   ├── results_processing.py   # Script for processing and visualizing results
│   └── results_histogram.py    # Script for generating histograms of results
│
└── README.md                   # Project overview and instructions
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Modeling-Electricity-Consumption-Patterns-in-Buildings-Using-Probability-Distribution-Functions.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Modeling-Electricity-Consumption-Patterns-in-Buildings-Using-Probability-Distribution-Functions
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Fitting Distributions

To fit a single distribution to the data, use the `fitOneDistDataCategory.py` script:
```sh
python src/fitOneDistDataCategory.py [dataset_set] [dataset_i] [op_filtro] [dist_name]
```

To fit multiple distributions to the data, use the `fitDistributionsDataCategory.py` script:
```sh
python src/fitDistributionsDataCategory.py [dataset_set] [dataset_i] [op_filtro]
```

### Processing Results

To process and visualize the results, use the `results_processing.py` script:
```sh
python src/results_processing.py
```

To generate histograms of the results, use the `results_histogram.py` script:
```sh
python src/results_histogram.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.

## Authors

- Mathias de Schietere de Lophem [GitHub](https://github.com/15154)
- Lambert Misselyn [GitHub](https://github.com/lmisselyn)
- Rosana Veroneze [GitHub](https://github.com/rveroneze)
- Hélène Verhaeghe [GitHub](https://github.com/363734)
- Axel Legay