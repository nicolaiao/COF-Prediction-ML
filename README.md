# COF-Prediction-ML

This repository contains all the code and resources related to my master's thesis on predicting the Coefficient of Friction (COF) from molecular descriptors using machine learning techniques. The project aims to enhance the prediction accuracy of COF by employing various machine learning models and addressing challenges such as data imbalance and feature selection.

## Contents

- `Data/`: Contains the dataset used for training and testing the models, along with the preprocessed data files.
- `Reports/`: Generated reports, including the profiling reports from EDA before and after preprocessing.
- `Models/`: Serialized machine learning models trained during the study.
- 
## Interactive Visualizations

- [View the interactive plot](./features_vs_cof.html): Interactive visualization of selected features against the Coefficient of Friction (COF).
- [View the profiling report from EDA before preprocessing](./profiling_data.html): Detailed exploratory data analysis report before any preprocessing.
- [View the profiling report from EDA after preprocessing](./profiling_data_after.html): Detailed exploratory data analysis report after data cleaning and preprocessing.

## Usage

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/COF-Prediction-ML.git
2. Navigate to the project directory:
   '''bash
   cd COF-Prediction-ML
3. Create and activate a virtual environment:
   '''bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
4. Install the required dependencies:
   '''bash
   pip install -r requirements.txt
5. Run the Jupyter notebooks:
   '''bash
   jupyter notebook

## Models Used
The following machine learning models were explored in this study:
- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Regression (SVR)
- AdaBoost
- XGBoost
- Multilayer Perceptron (MLP)
- Recurrent Neural Networks (RNN)
- Convolutional Neural Networks (CNN)
Each model's performance was evaluated based on its ability to accurately predict the COF from molecular descriptors. Detailed results and analysis can be found in the notebooks and reports.

## Future Work
Future enhancements to this project could include:

Incorporating more advanced molecular descriptors that account for complex chemical interactions.
Exploring additional machine learning models and ensemble techniques.
Integrating real-time monitoring data to improve model predictions.
Extending the dataset with more diverse lubricant samples to enhance generalizability.

## Acknowledgments
I would like to thank my supervisor, Prof. N. Espallargas, for her guidance and support throughout this project.

For any questions or feedback, please get in touch with me at nicolaiolsen982@gmail.com.
