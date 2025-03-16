# Titanic-Survival-Prediction

## Project Overview
This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset contains details such as age, gender, ticket class, and fare, which are used to build a predictive model.

## Dataset
The dataset used for this project is the Titanic dataset from Kaggle. It consists of:
- `PassengerId`: Unique ID given to each passenger
- `Survived`: Survival (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Sex of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of Siblings/Spouse aboard
- `Parch`: Number of Parent/Child aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare (Bristish Pound)
- `Cabin`: Cabin alloted to the passenger
- `Embarked`: Passenger from which port (S: Southampton, C: Cherbourg, Q: Queenstown)

Technologies Used
- Programmming Lnaguage - Python
- Pandas – Data manipulation
- NumPy – Numerical computations
- Matplotlib & Seaborn – Data visualization
- Scikit-Learn – Machine learning models

## Exploratory Data Analysis (EDA)
- Analyzed missing values and handled them appropriately.
- Visualized survival rates based on gender, class, and age.
- Identified important features influencing survival.

## Model Building
- Converted categorical data into numerical format
- Scaled numerical features

## Algorithms Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## Model Evaluation
- Used accuracy, precision, recall, and F1-score to evaluate models.
- Performed hyperparameter tuning to improve performance.

## Results and Conclusion
- The best-performing model was XGBoostClassifier with an accuracy of 83.006%.
- Gender, passenger class, and age were the most important features influencing survival.


## Future Improvements
- Try deep learning models for better accuracy.
- Use feature engineering techniques to enhance model performance.
- Apply ensemble methods to improve predictions.

## Acknowledgments
- The dataset is provided by Kaggle.
- This project is inspired by Titanic survival prediction competitions.
