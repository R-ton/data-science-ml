import os
from numpy import matrix, ones, array, reshape, concatenate, arange
from linear_regression import LinearRegression
from data_process import read_data_file
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, grid, title, xlabel, ylabel, scatter, show, figure, legend

def exp_simple_linear_regression(X,y):
    model_line_reg = LinearRegression(True)
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.25)
    model_line_reg.train(X_train, y_train)
    y_test_predict = model_line_reg.predict(X_test)
    y_train_predict = model_line_reg.predict(X_train)
    
    #Plotting
    figure('Simple linear regression')
    # Scatter plot the test data, using green color, with specified line widths and label
    scatter(X_test,y_test, c='g', linewidths=2, label='Trained PER for FG')
    # Plot the test data and the predicted values, using a specific color and label
    plot(X_test, y_test_predict, color='#455faa', label='Predicted PER for trained FG')
    xlabel('Actual FG%')
    ylabel('Predicted PER')
    title('Actual FG% vs Predicted PER')
    legend()


def exp_multiple_linear_regression(X, y):
    model_line_reg = LinearRegression(True)
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.25)
    model_line_reg.train(X_train, y_train)
    y_test_predict = model_line_reg.predict(X_test)
    y_train_predict = model_line_reg.predict(X_train)

    # Plotting
    figure('Multiple linear regression', figsize=(12, 6))
    # Create a scatter plot using y_test as x values and y_test_predict as y values
    scatter(y_test, y_test_predict, label='Predicted Test')
    # Plot a line from the minimum to the maximum values of y_test and y_test_predict
    plot([y_test.min(), y_test.max()], [y_test_predict.min(), y_test_predict.max()], 'k--', lw=2, label='best fit')
    xlabel('Actual Player in Dataset Index')
    ylabel('Predicted  Player in Dataset Index')
    title('Actual vs Predicted player in DataSet Index')
    legend(loc='best')



   
if __name__ == "__main__":
    data = read_data_file()
    X_multi_feature = data[['PPG','RPG', 'APG', 'FG%', 'PER']].values
    y_multi_target = arange(0, len(X_multi_feature))
    X_simple_feature = data['FG%'].values
    y_simple_target = data['PER'].values

    #predict the target PER for a feature FG%
    exp_simple_linear_regression(X=X_simple_feature, y=y_simple_target)

    #predict the player based on PPG','RPG', 'APG', 'FG%', 'PER
    exp_multiple_linear_regression(X=X_multi_feature, y=y_multi_target)

    show()