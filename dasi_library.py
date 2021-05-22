# DASI
#
# Think Create Learn 2018
#
# Not an alternative to using Pandas, Numpy, etc directly, but provides simpler learning base which can extend to full scipy stack later

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

# Pre-processing
from sklearn.preprocessing import MinMaxScaler

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

# Model prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# For viewing decision trees
from sklearn import tree

import warnings
warnings.simplefilter("ignore", FutureWarning)

# Read and write from CSV
############################################################################################
    
def readCsv(filename):
    '''Read from a csv file and create a DataFrame'''
    return pd.read_csv(filename)

def floatOnly(x):
    try:
        converted_value = float(x)
    except ValueError:
        converted_value = 'NaN'
    return converted_value

def readCsvFloatOnly(filename):
    '''Read from a csv file and create a DataFrame.  Only allow floats  NEED TO TEST THIS'''
    columns = df.columns.tolist()
    d = {}
    for col in columns:
        d[col] = floatOnly   
    return pd.read_csv(filename, converters=d)

def writeCsv(df, filename):
    '''Write the DataFrame to a csv file'''
    df.to_csv(filename, index=False)


# Selection Operations
############################################################################################
# Don't forget you can also select by row/col name and row/col index:
# dataFrame.loc[<ROWS RANGE> , <COLUMNS RANGE>]
# dataFrame.iloc[<ROWS INDEX RANGE> , <COLUMNS INDEX RANGE>]

def selectCol(df, column):
    '''Return just one column.  Provide the column name.'''
    return df[[column]]
    
def selectCols(df, columns):
    '''Return a subset of columns.  Provide a list of column names.'''
    return df[columns] 
    
def selectRows(df, condition):
    '''Return a subset of rows.  Provide a selection condition.'''
    return df.loc[condition]

def selectRowsWithNull(df, column):
    '''Select rows that contain null in the given column'''
    return df[df[column].isnull()]

def listUnique(df, column):
    '''List unique values in column'''
    return df[column].unique().tolist()

def listNonNumeric(df, column):
    '''List non numeric values in column'''
    return df[~df[column].str.isnumeric()][column].unique()

def listColumns(df):
    '''List out all column names'''
    return df.columns.tolist()

def selectFeaturesWithFewNullsPercent(df, percent):
    '''Select features with nulls <= percent %'''
    percents = df.isnull().mean()
    colNames = percents[percents<=percent].index.tolist()
    return df[colNames]   

def selectFeaturesWithAtLeastCount(df, count):
    '''Select features with count >= count'''
    counts = df.count()
    colNames = counts[counts>=count].index.tolist()
    return df[colNames] 

def renameColumn(df, oldName, newName):
    '''Rename a single column'''
    df.rename(columns={oldName: newName}, inplace=True)    

def groupByCount(df, groupCols, countCol):
    '''Group by the cols and count rows'''
    df1 = df.groupby(groupCols)[countCol].count().reset_index()
    return df1   

def flattenRowMultiIndex(df):
    df1 = df.copy()
    numLevels = len(df.index.levels)
    # Go through the levels and create new columns, one for each level
    for level in range(numLevels):
        df1[df1.index.names[level]+"_"] = df1.index.levels[level][df1.index.labels[level]]    
    return df1

# Cleansing and Feature Engineering
############################################################################################

def checkForNulls(df):
    '''Return the percentage of nulls in each column'''
    return df.isnull().mean()

def removeCol(df,column):
    '''Remove a single column from the DataFrame.  Provide the column name.'''
    #return df.loc[:, df.columns != column]
    return df.drop([column], axis=1)

def removeCols(df,columns):
    '''Remove multiple columns from the DataFrame.  Provide a list of column names.'''
    '''
    rv = df
    for col in columns:
        rv = rv.loc[:, rv.columns != col]
    return rv
    '''
    return df.drop(columns, axis=1)

def dropNullCols(df):
    '''Remove all cols containing nulls'''
    return df.dropna(axis=1)


def removeRowsByIndex(df, index):
    '''Remove rows with the given indices'''
    # Index can be a single index or list of indices
    return df.drop(index)

def removeRowsByColumnAndValue(df, column, value):
    '''Remove rows that contain the given value for the given column'''
    return df.drop(df[df[column]==value].index)

def removeRowsWithNullColumnValue(df, column):
    '''Remove rows where the value in column is null'''
    return df[df[column].notna()]



def dropNullRows(df):
    '''Remove all rows containing nulls'''
    return df.dropna()


def imputeNullWithValue(df, column, value):
    '''Replace nulls in the column with the value'''
    df1 = df.copy()
    df1[column] = df1[column].fillna(value=value)  # replace nulls with the value
    return df1

def imputeNullWithMean(df, column):
    '''Replace nulls in the column with the mean for the column'''
    df1 = df.copy()
    mean = df1[column].mean()                    # calculate the mean for the column
    df1[column] = df1[column].fillna(value=mean)  # replace nulls with the mean
    return df1

def imputeNullWithMedian(df, column):
    '''Replace nulls in the column with the median for the column'''
    df1 = df.copy()
    median = df1[column].median()                    # calculate the median for the column
    df1[column] = df1[column].fillna(value=median)    # replace nulls with the median
    return df1

def imputeNullWithMode(df, column):
    '''Replace nulls in the column with the mode for the column'''
    df1 = df.copy()
    mean = df1[column].mode()                    # calculate the mode for the column
    df1[column] = df1[column].fillna(value=mean)  # replace nulls with the mode
    return df1

def imputeNullWithMeanByGroup(df, column, groupby):
    '''Replace nulls in the column with the mean for the column within each group'''
    df1 = df.copy()
    df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(x.mean()))
    return df1

def imputeNullWithMedianByGroup(df, column, groupby):
    '''Replace nulls in the column with the median for the column within each group'''
    df1 = df.copy()
    df[column] = df.groupby(groupby)[column].transform(lambda x: x.fillna(x.median()))
    return df1

def imputeNullWithMeanForAllNumericColumns(df):
    '''Replace nulls in all numerical columns with the mean for the column'''
    df1 = df.copy()
    means = df1.mean().to_dict()
    for m in means:
        df1[m] = df1[m].fillna(value=means[m])
    return df1

def setValue(df, rowIndex, column, value):
    '''Set the value of a cell referenced by a row index and column name'''
    df.loc[rowIndex, column] = value

def oneHotEncode(df, columns, dummy_na=False):
    '''One-hot encode the column.  If dummy_na is True, add a column for nulls'''
    dummies = pd.get_dummies(df, columns=columns, dummy_na=dummy_na)
    return dummies


def splitFeatureOnSeparator(df, column, sep, newcols):
    '''Split column on sep into new columns named by the list newcols'''
    df1 = df.copy()
    df1[newcols] = df1[column].str.split(sep, expand=True)
    df1 = df1.drop([column], axis=1)
    return df1

def splitFeatureOnPosition(df, column, pos, newcols):
    '''Split a column at a given position into new columns named by the list newcols'''
    df1 = df.copy()
    df1[newcols[0]] = df1[column].str[:pos]
    df1[newcols[1]] = df1[column].str[pos:]
    df1 = df1.drop([column], axis=1)
    return df1

def splitFeatureDate(df, column):
    '''Split a date and time into its components'''
    df1 = df.copy()
    df1[column+"_day"] = df1[column].dt.day
    df1[column+"_month"] = df1[column].dt.month
    df1[column+"_year"] = df1[column].dt.year
    df1[column+"_weekday"] = df1[column].dt.weekday
    df1[column+"_hour"] = df1[column].dt.hour
    df1[column+"_minute"] = df1[column].dt.minute
    df1 = df1.drop([column], axis=1)
    return df1

def replaceValues(df, column, findValues, replaceWithValue):
    '''Find and replace values in a column.  Whole cell values.'''
    df1 = df.copy()
    df1[column] = df1[column].replace(findValues, replaceWithValue)
    return df1

def replaceSubstrings(df, column, findSubstrings, replaceWithSubstring):
    ''''Find and replace substrings in a column'''
    df1 = df.copy()
    for char in findSubstrings:
        df1[column] = df1[column].apply(lambda x: x.replace(char, replaceWithSubstring) if type(x) is str else x)
    return df1

def convertToDateTime(df, column):
    return pd.to_datetime(df[column])

def convertToFloat(df, column):
    return df[column].astype('float')

def convertToInt(df, column):
    return df[column].astype('float').astype('Int64') # convert to float first, then Int64 (new Pandas int type that supports NaNs)

def appendClass(df, class_name, feature, bins, labels):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    n = pd.cut(df[feature], bins = bins, labels=labels)
    c = df.copy()
    c[class_name] = n
    return c
    
def appendEqualCountsClass(df, class_name, feature, num_bins, labels):
    '''Append a new class feature named 'class_name' based on a split of 'feature' into classes with equal sample points.  Class names are in 'labels'.'''
    percentiles = np.linspace(0,100,num_bins+1)
    bins = np.percentile(df[feature],percentiles)
    n = pd.cut(df[feature], bins = bins, labels=labels, include_lowest=True)
    c = df.copy()
    c[class_name] = n
    return c

        

# Pivotting and unpivotting
############################################################################################
    
def pivot(df, values, ids, columns):
    '''Pivot the data, with 'values' in the cells, 'ids' as the columns to remain in place and 'columns' as the columns to be pivotted.'''
    v = pd.pivot_table(df, values=values, index=ids,columns=columns)
    v = v.reset_index()
    v.columns.name = None
    return v
    
def unpivot(df, ids):
    '''Unpivot the data. ids are the columns to remain in place '''
    return pd.melt(df, id_vars=ids, value_vars=df.columns.drop(ids)) 


# Merging
############################################################################################

def mergeOn(df, other, on, to):
    '''Join one dataset to another'''
    return df.merge(other, left_on=on, right_on=to, how='left')


# Prepare data for machine learning
############################################################################################

def splitXY(df, columnName):
    '''Split dataset into input and target features'''
    Y = df[columnName]
    X = df.loc[:, df.columns != columnName]
    return (X,Y)
    

def classDistribution(df, class_name):
    '''Return the class distribution, i.e. the number of rows against each class.'''
    #alternatively df[class_name].value_counts()
    return df.groupby(class_name).size()

def rescale(df):
    '''Perform rescale preprocessing to scale each feature into the range [0,1]'''
    scaler = MinMaxScaler(feature_range=(0,1))
    rv = df.copy()
    rv[df.columns] = scaler.fit_transform(df[df.columns])
    return rv

def selectFeaturesKBestClassification(k, X, Y):
    '''Use univariate selection to select the k best features for a classification task'''
    return _selectFeaturesKBest(k, X, Y, chi2)

def selectFeaturesKBestRegression(k, X, Y):
    '''Use univariate selection to select the k best features for a regression task'''
    return _selectFeaturesKBest(k, X, Y, f_regression)    
    
def _selectFeaturesKBest(k, X, Y, score_func):
    '''Use univariate selection to select the k best features'''
    test = SelectKBest(score_func=score_func, k=k)
    #fit = test.fit(X,Y.iloc[:,0].tolist())
    fit = test.fit(X,Y.tolist())
    #set_printoptions(precision=3)
    #print(fit.scores_)
    features = fit.transform(X)
    #print(features[0:k+1,:])
    selected_cols = test.get_support()
    #print(selected_cols)        
    
    # Now pull out the feature names that match
    selected_features = []
    for i in range(len(selected_cols)):
        if selected_cols[i]:
            selected_features.append(X.columns[i])
    #print(selected_features)   
    
    return X[selected_features]


# Build machine learning model
############################################################################################

def trainTestSplit(X, Y, test_size, random_state):
    '''Split X and Y into test and training data.'''
    X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return (X_train, X_test, Y_train.values.ravel(), Y_test.values.ravel())

def modelFit(X, Y, algorithm):
    '''Use the data to fit (i.e. train) the model'''
    model = algorithm()
    model.fit(X, Y)
    return model

def predict(model, X):
    '''Use the model to make predictions about input features in X'''
    return model.predict(X)

def evaluateAlgorithmsClassification(X, Y, algorithms, random_state):
    '''Evaluate all provided algorithms against the training data'''
    results = []

    # Split into 10 folds
    kfold = KFold(n_splits=10, random_state=random_state)

    # Perform 10-fold CV on all algorithms
    for algorithm in algorithms:
        cv_results = cross_val_score(algorithm(), X, Y, cv=kfold, scoring='accuracy')
        results.append((algorithm.__name__, cv_results.mean()))
    return results

def evaluateAlgorithmsRegression(X, Y, algorithms, random_state):
    '''Evaluate all provided algorithms against the training data'''
    results = []

    # Split into 10 folds
    kfold = KFold(n_splits=10, random_state=random_state)

    # Perform 10-fold CV on all algorithms
    for algorithm in algorithms:
        cv_results = cross_val_score(algorithm(), X, Y, cv=kfold, scoring='neg_mean_absolute_error') #scoring='neg_mean_squared_error')
        results.append((algorithm.__name__, -cv_results.mean()))
    return results
  
def comparePredictionsWithOriginals(df, predictions, original):
    '''Compare results from the model with the orginal (correct) results'''
    df = df.copy()
    df['Prediction'] = predictions
    df['Original'] = original
    return df

        
# Visualisation
############################################################################################
    
def boxPlotAll(df):
    '''Show box plots for each feature'''
    data_cols = len(df.columns)
    unit_size = 5
    layout_cols = 4
    layout_rows = int(data_cols/layout_cols+layout_cols)
    df.plot(kind='box', subplots=True, figsize=(layout_cols*unit_size,layout_rows*unit_size), layout=(layout_rows,layout_cols))
    pyplot.show()        

def histPlotAll(df):
    '''Show histograms for each feature'''
    data_cols = len(df.columns)
    unit_size = 5
    layout_cols = 4
    layout_rows = int(data_cols/layout_cols+layout_cols)
    df.hist(figsize=(layout_cols*unit_size,layout_rows*unit_size), layout=(layout_rows,layout_cols))
    pyplot.show()     

def bubbleChart(df, xCol, yCol, sizeCol, colourCol, labelsCol, labelsToShow=None, minBubble=5, maxBubble=5000):
    '''Display a scatter plot with attributes on the x axis, y axis, circle area and circle colour'''

    categorical = df[colourCol].dtype == np.dtype('O')

    # Get x and y values
    x = df[xCol]
    y = df[yCol]
    
    # Get the circle area values and scale the area values to sensible circle areas
    '''
    areaScaled = df[sizeCol]
    minArea = areaScaled.min()
    maxArea = areaScaled.max()
    rescaled = (areaScaled-minArea) / (maxArea - minArea) 
    area = np.pi * (30 * rescaled) **2         # 0 to 15 point radiuses
    area = area.where(area>2,2)             # min circle area
    '''

    # Scale the size values to a sensible circle size range
    area = df[sizeCol]
    area = np.interp(area, (area.min(), area.max()), (minBubble, maxBubble))

    
    # Get the circle colour values
    colours = df[colourCol]

    # Create the colour map
    if categorical:
        keys=df[colourCol].unique()
        cm = pyplot.cm.get_cmap('tab10')
        dictionary = dict(zip(keys, cm.colors))
        
    # Create the scatter plot
    fig = pyplot.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    if categorical:
        pyplot.scatter(x, y, s=area, alpha=0.5, c=colours.apply(lambda x : dictionary[x]), label=None)
    else:        
        pyplot.scatter(x, y, s=area, alpha=0.5, c=colours)
    
    # Display the labels
    labels = df[labelsCol]    
    for i in range(len(labels)):
        label = labels[i]
        if labelsToShow is None or label in labelsToShow:
            xcoord = x[i]
            ycoord = y[i]
            ax.annotate(label, xy=(xcoord,ycoord), xytext=(xcoord,ycoord),arrowprops=dict(facecolor='black', shrink=0.05),)

    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
                'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'14'}

    # Add legend
    if categorical:        
        for colour in colours.unique() :
            pyplot.scatter([], [], c=dictionary[colour], alpha=0.6, s=200,
                        label=colour)
        pyplot.legend(frameon=False, labelspacing=2, title=colourCol)
    else:
        pyplot.colorbar()

    # Plot it
    pyplot.title(xCol + " vs " + yCol + ", size=" + sizeCol + ", colour=" + colourCol, title_font)
    pyplot.xlabel(xCol,axis_font)
    pyplot.ylabel(yCol,axis_font)
    pyplot.show()             

def scatterMatrix(df):
    '''Show a scatter matrix of all features.'''
    unit_size = 5
    pd.plotting.scatter_matrix(df,figsize=(unit_size*4, unit_size*4))
    pyplot.show()
        
def correlationMatrix(df):
    '''Show a correlation matrix for all features.'''
    columns = df.select_dtypes(include=['float64','int64']).columns
    df1 = df[columns]
    fig = pyplot.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df1.corr(), vmin=-1, vmax=1, interpolation='none',cmap='RdYlGn')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)
    pyplot.show()       
        
def classComparePlot(df, class_name, plotType='density'):
    '''Show comparative plots comparing the distribution of each feature for each class.  plotType can be 'density' or 'hist' '''
    numcols = len(df.columns) - 1
    
    unit_size = 5
    classes = df[class_name].nunique()           # no of uniques classes
    class_values = df[class_name].unique()       # unique class values
    
    print('Comparative histograms for',class_values)
    
    colors = pyplot.cm.get_cmap('tab10').colors
    fig = pyplot.figure(figsize=(unit_size*4,numcols*unit_size))
    ax = [None]*numcols 
    i = 0
    for col_name in df.columns:
        minVal = df[col_name].min()
        maxVal = df[col_name].max()
    
        if col_name != class_name:                
            ax[i] = fig.add_subplot(numcols,1,i+1)   
            for j in range(classes):   
                selectedCols = df[[col_name,class_name]]
                filteredRows = selectedCols.loc[(df[class_name]==class_values[j])]
                values = filteredRows[col_name]
                values.plot(kind=plotType,ax=ax[i],color=[colors[j]], alpha = 0.8, label=class_values[j])
                ax[i].set_title(col_name)
                ax[i].grid()                                  
                #(df[[col_name,class_name]].loc[(df[class_name]==class_values[j])])[[col_name]].hist(ax=ax[i],color=[colors[j]], alpha = 0.5, label=class_values[j])
            ax[i].legend()
            i += 1
    pyplot.show()
        

# Model Inspection
############################################################################################

def logisticRegressionSummary(model, column_names):
    '''Show a summary of the trained logistic regression model'''

    # Get a list of class names
    numclasses = len(model.classes_)
    if len(model.classes_)==2:
        classes =  [model.classes_[1]] # if we have 2 classes, sklearn only shows one set of coefficients
    else:
        classes = model.classes_

    # Create a plot for each class
    for i,c in enumerate(classes):
        # Plot the coefficients as bars
        fig = pyplot.figure(figsize=(8,len(column_names)/3))
        fig.suptitle('Logistic Regression Coefficients for Class ' + str(c), fontsize=16, y=1.08)
        rects = pyplot.barh(column_names, model.coef_[i],color="lightblue")
        
        # Annotate the bars with the coefficient values
        for rect in rects:
            width = round(rect.get_width(),4)
            pyplot.gca().annotate('  {}  '.format(width),
                        xy=(0, rect.get_y()),
                        xytext=(0,2),  
                        textcoords="offset points",  
                        ha='left' if width<0 else 'right', va='bottom')        
        pyplot.show()
        #for pair in zip(X.columns, model_lr.coef_[i]):
        #    print (pair)

def decisionTreeSummary(model, column_names):
    '''Show a summary of the trained decision tree model'''

    # Plot the feature importances as bars
    fig = pyplot.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Decision tree feature importance', fontsize=16, y=1.08)
    rects = pyplot.barh(column_names, model.feature_importances_,color="khaki")

    # Annotate the bars with the feature importance values
    for rect in rects:
        width = round(rect.get_width(),4)
        pyplot.gca().annotate('  {}  '.format(width),
                    xy=(width, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left', va='bottom')    

    pyplot.show()

def linearRegressionSummary(model, column_names):
    '''Show a summary of the trained linear regression model'''

    # Plot the coeffients as bars
    fig = pyplot.figure(figsize=(8,len(column_names)/3))
    fig.suptitle('Linear Regression Coefficients', fontsize=16, y=1.08)
    rects = pyplot.barh(column_names, model.coef_,color="lightblue")

    # Annotate the bars with the coefficient values
    for rect in rects:
        width = round(rect.get_width(),4)
        pyplot.gca().annotate('  {}  '.format(width),
                    xy=(0, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left' if width<0 else 'right', va='bottom')        
    pyplot.show()

    
def viewDecisionTree(model, column_names):
    '''Visualise the decision tree'''
    import graphviz 

    dot_data = tree.export_graphviz(model, out_file=None,
            feature_names=column_names,
            class_names=[str(c) for c in model.classes_],
            filled=True, rounded=True,
            special_characters=True)
    graph = graphviz.Source(dot_data) 
    return graph   


# Test data
############################################################################################

def testData1():
    euData = {
        'area'       : [357168, 551394, 244820, 301318, 498468],
        'population' : [81.5, 67.0, 65.1, 60.6, 46.4],
        'employment'  : [74.7, 64.6, 75.3, 57.2, 60.5]}
    return DataFrame(euData, index=["Germany", "France", "United Kingdom", "Italy", "Spain"])


