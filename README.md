# GRAB AI FOR S.E.A CHALLENGE [(Safety)](https://www.aiforsea.com/safety)

###  PROBLEM STATEMENT

Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, derive a model that can detect dangerous driving trips.

My entry for the competition.

### Notes
1. features and labels folders are empty in this repository, just download the file using browser or wget from the Grab's [repo of the file](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip) and extract, you will get the features and labels folder plus the data dictionary file.
2. utility and training models file are stored in the modules folder.
3. lightgbm and xgboost .dict files are stored (just run the optimize_lgb / optimize_xgb with this parameters to test the models.) to use my best-models.
the parameters are found using the [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package.
4. if you want to return the classifier models for xgboost and lightgbm just add `clf` variable to the return statement of the optimize_lgb / optimize_xgb when test_phase=True.


### Approach

1. aggregate the dataframe with original features (with second)
2. aggregate the dataframe with minutes (second converted to minute) and merge with original aggregated dataframe using bookingid
3. train models on the whole feature set with XGB and LGB
4. use bayesian optimization to find the optimal parameters
5. retrain the model using the found parameters stored in .dict file (for xgb and lgb)
6. predict and check results.

### How to use the notebook [safety_for_sea_rafael_macalaba.ipynb](https://github.com/rafmacalaba/grab_safety/blob/master/src/safety_for_sea_rafael_macalaba.ipynb)
Step by step:

1. instantiate the variables for labels and features
2. run get_all_data to get the initial dataframe
3. run process_all_data to get X_train, X_test, y_train, y_test
4. run xgb_optimize and lgb_optimize
5. run get_test_result to get the hold out evaluation (using X_test as your hold out data and y_test as your ground truth data.

Additional steps for grab evaluator:

6. run get_all_data for your files for the initial dataframe
7. run process_all_data and set test_grab_evaluation=True to get your final process dataframe
8. run get_test_result to get the evaluation score of your files (using run_process_all_data output as your hold out data and your file(s) ground truth data.

### Models and Library used.

[xgboost](https://xgboost.readthedocs.io/en/latest/)

[lighgbm](https://lightgbm.readthedocs.io/en/latest/)

[BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
