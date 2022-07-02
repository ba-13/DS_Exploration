import sys  # default library
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
from sklearn.linear_model import LinearRegression
import random  # default library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings  # default library

"""
This prediction involves Linear regression fitted on aptly modified dataset given as `IPL_train.csv`.
The model will train on-spot, so please wait ~25 seconds for the whole file to run.
The major time is taken by data transformations done in `add_runs_wickets_cumulative_column`.
"""

warnings.filterwarnings("ignore")


def prediction(df: pd.DataFrame, match):
    """Prediction from Extrapolation of cumulative runs

    Args:
        df (pd.DataFrame): Dataframe
        match (int): match_id

    Returns:
        pred_needed: Predictions of 120 to 130 balls that innings
    """
    match_info = df.loc[df['match_id'] == match]
    till_over = np.floor(match_info['over'].iloc[-1])
    if till_over > 12:
        till_over = 8
        match_train = match_info.loc[match_info['over'] < till_over]
    else:
        match_train = match_info

    c = np.array(match_info['total_runs'])
    c_train = np.array(match_train['total_runs'])
    x = np.linspace(0, match_info.shape[0]-1, num=match_info.shape[0])
    x_train = np.linspace(0, match_train.shape[0]-1, num=match_train.shape[0])

    x_pred = []
    for i in range(10):
        x_pred.append(20*6+i)
    fit = np.polyfit(x_train, c_train, 1)
    line = np.poly1d(fit)
    pred_needed = line(x_pred)
    pred = line(x)
    return pred_needed, pred, c, c_train, x, x_train, till_over


def plot_pred(df: pd.DataFrame, match):
    pred_needed, pred, c, c_train, x, x_train, till_over = prediction(
        df, match)
    fig, axis = plt.subplots()
    axis.plot(x, c)
    axis.plot(x_train, c_train)
    axis.plot(x, pred)
    plt.show()


def fix_wicket_column(df: pd.DataFrame):
    df['wickets'] = df['wickets'].fillna(0)
    df.loc[df['wickets'] != 0, 'wickets'] = 1
    return df


def merge_over_and_ball_columns(df: pd.DataFrame):
    df['over'] = df['over'] + df['ball']*0.1
    df.drop(labels='ball', axis=1, inplace=True)
    return df


def add_runs_wickets_cumulative_column(df: pd.DataFrame, n: int):
    """Adds runs and wickets, cumulative and past few weeks, columns

    Args:
        df (pd.DataFrame): Dataframe
        n (int): till which week to consider
    """
    assert n < 8, "ValueError"
    matches = df['match_id'].unique()
    match_dataframes = []
    for match in matches:
        match_info = df[df['match_id'] == match].reset_index()
        runs = 0
        wickets = 0
        runs_past_n = []
        wickets_past_n = []
        runs_his = []
        wickets_his = []
        for idx in range(match_info.shape[0]):
            runs += match_info['total_runs'].iloc[idx]
            match_info['total_runs'].iloc[idx] = runs
            wickets += match_info['wickets'].iloc[idx]
            match_info['wickets'].iloc[idx] = wickets
            runs_past_n.append(match_info['total_runs'].iloc[idx])
            wickets_past_n.append(match_info['wickets'].iloc[idx])
            if len(runs_past_n) > n*6:
                runs_past_n.pop(0)
                wickets_past_n.pop(0)
            runs_his.append(runs_past_n[-1]-runs_past_n[0])
            wickets_his.append(wickets_past_n[-1]-wickets_past_n[0])
        pred_needed, pred, _, _, _, _, till_over = prediction(
            match_info, match)
        match_info = match_info.assign(
            runs_past=pd.Series(np.array(runs_his)).values)
        match_info = match_info.assign(
            wickets_past=pd.Series(np.array(wickets_his)).values)
        match_info = match_info.assign(
            Total=pd.Series(np.ones(len(runs_his))*runs).values)
        match_info = match_info.assign(run_rate_pred=pd.Series(np.ones(len(
            runs_his))*pred_needed[0]).values)  # pred_needed[0] is prediction at 20*6 balls
        match_dataframes.append(match_info)
    df = pd.concat(match_dataframes).reset_index()
    return df


def transform_data(df: pd.DataFrame):
    df = df.drop(labels=['batsman', 'non_striker', 'bowler', 'wide_runs', 'bye_runs', 'legbye_runs',
                         'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs', 'player_dismissed', 'fielder'], axis=1)
    df.rename(columns={'dismissal_kind': 'wickets'}, inplace=True)
    matches = df['match_id'].unique()
    df = fix_wicket_column(df)
    df = merge_over_and_ball_columns(df)
    df = add_runs_wickets_cumulative_column(df, 4)
    df.drop(labels=['level_0', 'index'], axis=1, inplace=True)
   #  encoded_df = pd.get_dummies(data=df, columns=['batting_team', 'bowling_team'])
    # Total column contains the total runs of that match that's provided in the data
    X = df.drop(labels=['Total'], axis=1)
    y = df['Total'].values
    return X


def predict_score(batting_team, bowling_team, over, total_runs, wickets, runs_past, wickets_past, run_rate_pred):
    """Predicts score given parameters and model

    Args:
        batting_team (string): Batting Team
        bowling_team (string): Bowling Team
        over (int): Overs till now
        total_runs (int): Total runs till now
        wickets (int): Wickets till now
        runs_past (int): total runs past 4 overs
        wickets_past (int): total wickets past 4 overs
        run_rate_pred (int): prediction of final score using run rate
    """

    temp_array = list()

    # Batting Team
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Deccan Chargers':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Gujarat Lions':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kochi Tuskers Kerala':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == 'Pune Warriors':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == 'Rising Pune Supergiants':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Deccan Chargers':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Gujarat Lions':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kochi Tuskers Kerala':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == 'Pune Warriors':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == 'Rising Pune Supergiants':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Overs, Runs, Wickets, Runs_in_prev_7
    temp_array = [over, total_runs, wickets, runs_past,
                  wickets_past, run_rate_pred] + temp_array

    # Converting into numpy array
    temp_array = np.array([temp_array])

    # Prediction
    return int(linear_regressor.predict(temp_array)[0])


def randomly_sample_and_show_from_train(df: pd.DataFrame, match_id: int = -1, print_=True):
    """This function is a helper function to for visualisation of performance of the model.

    Args:
        df (pd.DataFrame): Dataframe
        match_id (int, optional): if -1, will be randomly sampled. Defaults to -1.
        print_ (bool, optional): To print details of that particular match. Defaults to True.

    """
    matches = df['match_id'].unique()
    match = match_id if match_id != -1 else random.choice(matches)
    match_info = df.loc[df['match_id'] == match].reset_index()
    value = match_info['Total'][0]

    consideringTillBall = 42

    val = predict_score(match_info['batting_team'][0], match_info['bowling_team'][0], over=match_info['over'][consideringTillBall], total_runs=match_info['total_runs'][consideringTillBall], wickets=match_info['wickets']
                        [consideringTillBall], runs_past=match_info['runs_past'][consideringTillBall], wickets_past=match_info['wickets_past'][consideringTillBall], run_rate_pred=match_info['run_rate_pred'][consideringTillBall])

    pred_needed, pred, c, c_train, x, x_train, till_over = prediction(
        df, match)

    if print_:
        print("Match ID:", match_info['match_id'][0])
        print("Wickets:", match_info['wickets'].iloc[till_over*6])
        print("Actual prediction from extrapolation:", round(pred[-1], 2))
        print("Prediction from Model:", val)
        print("Actual Value:", value)
        plot_pred(df, match_info['match_id'][0])
    return val, round(pred[-1], 2), match_info['wickets'].iloc[till_over*6], value


def choose_from_testset(df_transformed: pd.DataFrame, match_id: int = -1, print_=True):
    matches = df_transformed['match_id'].unique()
    match = match_id if match_id != -1 else random.choice(matches)
    match_info = df_transformed.loc[df_transformed['match_id'] == match].reset_index(
    )

    consideringTillBall = match_info.shape[0]-1

    val = predict_score(match_info['batting_team'][0], match_info['bowling_team'][0], over=match_info['over'][consideringTillBall], total_runs=match_info['total_runs'][consideringTillBall], wickets=match_info['wickets']
                        [consideringTillBall], runs_past=match_info['runs_past'][consideringTillBall], wickets_past=match_info['wickets_past'][consideringTillBall], run_rate_pred=match_info['run_rate_pred'][consideringTillBall])

    pred_needed, pred, c, c_train, x, x_train, till_over = prediction(
        df_transformed, match)

    if print_:
        print("Match ID:", match_info['match_id'][0])
        print("Wickets:", match_info['wickets'].iloc[-1])
        print("Actual prediction from extrapolation:",
              round(pred_needed[0], 2))
        print("Prediction from Model:", val)
        plot_pred(df_transformed, match_info['match_id'][0])
    return val, round(pred_needed[0], 2), match_info['wickets'].iloc[-1]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Not providing any paths...")
        print("Using default paths.")
        print()
        path_train = "./IPL_train.csv"
        path_test = "./IPL_test.csv"
    else:
        path_train = sys.argv[1]
        path_test = sys.argv[2]
    df = pd.read_csv(path_train)

    df = df.drop(labels=['batsman', 'non_striker', 'bowler', 'wide_runs', 'bye_runs', 'legbye_runs',
                         'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs', 'player_dismissed', 'fielder'], axis=1)
    df.rename(columns={'dismissal_kind': 'wickets'}, inplace=True)
    df = fix_wicket_column(df)
    df = merge_over_and_ball_columns(df)
    # 4 is the number of past overs to consider, after trials
    df = add_runs_wickets_cumulative_column(df, 4)
    df.drop(labels=['level_0', 'index'], axis=1, inplace=True)
    encoded_df = pd.get_dummies(
        data=df, columns=['batting_team', 'bowling_team'])

    X_train = encoded_df.drop(labels=['Total'], axis=1)[
        encoded_df['match_id'] < 500]
    X_test = encoded_df.drop(labels=['Total'], axis=1)[
        encoded_df['match_id'] >= 500]

    y_train = encoded_df[encoded_df['match_id'] < 500]['Total'].values
    y_test = encoded_df[encoded_df['match_id'] >= 500]['Total'].values

    X_train.drop(labels='match_id', axis=True, inplace=True)
    X_test.drop(labels='match_id', axis=True, inplace=True)

    print("Splitting train.csv into Training and Validation sets")
    print("Training set: {} and Validation set: {}".format(
        X_train.shape, X_test.shape))

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    y_pred_train = linear_regressor.predict(X_train)
    y_pred_lr = linear_regressor.predict(X_test)
    print("---- Linear Regression - Model Evaluation over Training Set ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_train, y_pred_train)))
    print("Mean Squared Error (MSE): {}".format(mse(y_train, y_pred_train)))
    print("Root Mean Squared Error (RMSE): {}".format(
        np.sqrt(mse(y_train, y_pred_train))))
    print("---- Linear Regression - Model Evaluation over Validation Set ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
    print("Root Mean Squared Error (RMSE): {}".format(
        np.sqrt(mse(y_test, y_pred_lr))))

    # Importing dataset
    df_test = pd.read_csv(path_test)
    df_transformed = transform_data(df_test)
    with open("predictions.csv", "w") as f:
        matches = df_transformed['match_id'].unique()
        f.write("match_id,prediction\n")
        for match in matches:
            print("Match ID:", match, end=" ")
            pred, extra_pred, wickets = choose_from_testset(
                df_transformed, match, print_=False)
            if wickets > 4:
                weight = 0.7
                final_pred = extra_pred*weight + pred*(1-weight)
            else:
                weight = 0.2
                final_pred = extra_pred*weight + pred*(1-weight)
            f.write(str(match) + "," + str(np.round(final_pred, 2))+"\n")
            print(", Pred_Model:", pred, ", Pred_extrapolation:",
                  extra_pred, ", Final_pred:", final_pred)
