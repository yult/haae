# Harmonizing Attention and Attention-Free Encoders with a Novel Fusion Strategy for Student Performance Prediction - Official Implementation


## Project Overview

This repository contains the official PyTorch implementation of the paper:Harmonizing Attention and Attention-Free Encoders with a Novel Fusion Strategy for Student Performance Prediction.

## Directory Structure

    Project Root/
    │
    ├── source/ # Raw data storage
    │ ├── assessments.csv
    │ ├── courses.csv
    │ ├── studentAssessment.csv
    │ ├── studentInfo.csv
    │ ├── studentRegistration.csv
    │ ├── studentVle.csv
    │ └── vle.csv
    ├── script/ # Python scripts
    │ ├── config.yaml # Configuration file
    │ ├── config_loader.py # Configuration loader script
    │ ├── data_merge.py # Sample generation script
    │ ├── DNN-G.py # Model DNN-G script
    │ ├── DNN-GC.py # Model DNN-GC script
    │ ├── DNN-ALL.py # Model DNN-ALL script
    │ ├── LSTM.py # Model LSTM script
    │ ├── WLF-ED.py # Model WLF-ED script
    │ ├── AWLF-ED.py # Model AWLF-ED script
    │ └── measurements_plot.py # Measurements plot script
    ├── sample/ # Intermediate files and generated samples
    │ ├── graphic_dict.csv # Graphic dict file
    │ ├── graphic_course_dict.csv # Graphic course dict file
    │ ├── static_dict.csv # Static dict file
    │ ├── dynamic_dict.csv # Dynamic dict file
    │ ├── dict.csv # Static and dynamic dict file
    │ └── measurements.csv # Model measurements results for plot
    ├── output/ # Generated files and final results
    └── README.md # Project documentation (this file)

## Requirements

- Python 3.9.19

Required packages:
- torch 2.0.0+cpu
- numpy 1.26.4
- pandas 1.2.4
- sklearn 1.4.2
- imblearn 0.12.2
- matplotlib 3.5.1
- seaborn 0.13.2
- yaml 6.0.2
- logging 0.5.1.2
- ctypes 1.1.0
- scipy 1.13.0
- filelock 3.13.1
- retrying 1.3.4

## Installation and Configuration

1. Clone or download this project to your local machine
2. Install required dependencies
3. Configure project paths:
   - Open the `script/config.yaml` file
   - Modify the `data_dir` value to reflect your actual project directory path:
     ```yaml
     data_dir: "Your absolute path to the project root directory"
     ```
   - Example: If the project is located at `E:/Projects/ML_Project`, configure as:
     ```yaml
     data_dir: "E:/Projects/ML_Project"
     ```

## Usage Instructions

Execute the following steps in sequential order:

1. Transfer the source data from the publicly accessible address to the directory `Project Root/source`, with detailed file information provided in the Directory Structure section.

2. Navigate to the script directory:
   ```bash
   cd {Project Root}/script
   ```
3. Generate sample data:
   ```bash
   python data_merge.py
   ```
4. Execute individual model or plot scripts:
   ```bash
   python DNN-G.py
   python DNN-GC.py
   python DNN-ALL.py
   python LSTM.py
   python WLF-ED.py
   python AWLF-ED.py
   python measurements_plot.py
   ```

## Output Results

Upon completion of all script executions, results will be saved in the following locations:
- Intermediate files: sample/ directory
- Final results: output/ directory

## License
This project is released under the MIT License.