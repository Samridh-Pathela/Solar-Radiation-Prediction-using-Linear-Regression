# Solar-Radiation-Prediction-using-Linear-Regression

----------About Dataset------------

The prediction model works on datasetscfrom the HI-SEAS weather station from four months (September through December 2016) between Mission IV and Mission V.

For each dataset, the fields are:

A row number (1-n) useful in sorting this export's results
The UNIX time_t date (seconds since Jan 1, 1970). Useful in sorting this export's results with other export's results
The date in yyyy-mm-dd format
The local time of day in hh:mm:ss 24-hour format
The numeric data, if any (may be an empty string)
The text data, if any (may be an empty string)

The units of each dataset are:

Solar radiation: watts per meter^2
Temperature: degrees Fahrenheit
Humidity: percent
Barometric pressure: Hg
Wind direction: degrees
Wind speed: miles per hour
Sunrise/sunset: Hawaii time

---------About Model/Algorithm-------

To quickly evaluate the performance of an algorithm on our problem we create a train and test
split of your dataset. The training dataset is used to prepare a model, to train it. We pretend the
test dataset is new data where the output values are withheld from the algorithm. We gather
predictions from the trained model on the inputs from the test dataset and compare them to the
withheld output values of the test set.
Comparing the predictions and withheld outputs on the test dataset allows us to compute a
performance measure for the model on the test dataset. This is an estimate of the skill of the
algorithm trained on the problem when making predictions on unseen data.
Using the train/test method of estimating the skill of the procedure on unseen data often has a
high variance. This means that when it is repeated, it gives different results. The outcome is
that we may be quite uncertain about how well the procedure actually performs on unseen data
and how one procedure compares to another.


---- About Code----

Sure! Let's go through the code step-by-step and explain each part in a simple manner:

Importing Libraries:
The code starts by importing some Python libraries that are needed for data analysis and visualization. pandas is used for data handling, matplotlib.pyplot for plotting graphs, and seaborn for enhanced visualizations.

Reading the Data:
The code reads a CSV file named "SolarPrediction.csv" using pd.read_csv() and stores it in a pandas DataFrame called df.

Exploring the Data:
The code displays the first few rows of the DataFrame using df.head() and provides a summary of the numerical data using df.describe(). This helps to get a quick understanding of the dataset.

Data Selection and Visualization:
The code selects three columns from the DataFrame: 'Temperature', 'Radiation', and 'Pressure', and stores them in a new DataFrame called cdf.

Histogram Visualization:
The code plots histograms of 'Temperature', 'Radiation', and 'Pressure' using cdf.hist(). A histogram shows the distribution of values in each column.

Scatter Plots:
The code creates scatter plots to visualize the relationship between 'Temperature' and 'Radiation' and between 'Pressure' and 'Radiation'. Scatter plots display individual data points as dots on a graph, helping us see if there is any correlation or pattern between the variables.

Data Preparation:
The code prepares the data for building a machine learning model. It separates the independent variables (features) and the dependent variable (target) for the model. The independent variables are 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', and 'Speed', stored in the DataFrame X. The dependent variable is 'Radiation', stored in the Series y.

Train-Test Split:
The code splits the data into training and testing sets using the train_test_split function from scikit-learn. This step is essential to evaluate the model's performance on unseen data.

Linear Regression Model Building:
The code creates a Linear Regression model using scikit-learn's LinearRegression class, and it fits the model to the training data using l.fit(X_train, y_train).

Model Evaluation:
The code evaluates the performance of the model using various metrics:

It prints the intercept of the linear regression model using l.intercept_.
It makes predictions on the test data using the trained model and calculates the sum of squared errors (sse) and total sum of squares (sst).
It calculates the R-squared value, which represents the proportion of the variance in the target variable that is predictable from the independent variables.
It visualizes the predictions against the actual values using scatter plots.
It checks for the normality of residuals and displays a histogram.
Metrics Calculation:
The code calculates three common evaluation metrics for regression models:
Mean Absolute Error (MAE): The average absolute difference between the predicted values and the actual values.
Mean Squared Error (MSE): The average of the squared differences between the predicted and actual values.
Root Mean Squared Error (RMSE): The square root of MSE, which gives a measure of the model's error in the original units of the target variable.
Overall, the code performs data exploration, visualization, prepares data for modeling, builds a Linear Regression model, evaluates its performance using various metrics, and presents the results visually. It's a good starting point for a simple machine learning project focusing on predicting solar radiation based on weather parameters.


