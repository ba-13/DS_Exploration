# Problem Statement 3

My approach was using Linear regression over aptly modified dataset, the dataset given is `IPL_train.csv`.

You need to run the file `final.py` with required dependencies being:

```txt
sklearn==1.1.1
numpy==1.23.0
matplotlib==3.5.2
pandas==1.4.3
```

To run the file, use the following command, given the dependencies are installed.

```bash
python3 final.py ./IPL_train.csv ./IPL_test.csv
```

You can replace `./IPL_train.csv` with your train file path and `./IPL_test.csv` with the test file path.  
Printed lines include the data regarding the model performance, as well as predictions from the model as well as extrapolation (explained later).  
The folder would now include `predictions.csv` which contains two columns: `match_id` being the match_id of the corresponding match given in the test dataset, `prediction` being the predicted score after `120` balls (fixing the number of balls per innings).

## Workings

Given the dataset contains columns:
match_id,batting_team,bowling_team,over,ball,batsman,non_striker,bowler,wide_runs,bye_runs,legbye_runs,noball_runs,penalty_runs,batsman_runs,extra_runs,total_runs,player_dismissed,dismissal_kind,fielder.

I dropped the columns:
'batsman', 'non_striker', 'bowler', 'wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs', 'player_dismissed', 'fielder'

This is considering that micro-events like which kind of runs can't be predicted if not given, also player names are dropped because we don't have the information of the players that will be involved in the game for the overs whose information isn't provided. This is also true for the players involved when a wicket is taken.

Then the code involves:

```py
df = fix_wicket_column(df)
df = merge_over_and_ball_columns(df)
# 4 is the number of past overs to consider, after trials
df = add_runs_wickets_cumulative_column(df, 4)
```

Each of these functions does data-preprocessing. `fix_wicket_column` converts `dismissal_kind` column to a column containing wicket-events. `merge_over_and_ball_columns` merges over and ball columns. `add_runs_wickets_cumulative_column` introduces 4 columns, 2 being cumulative runs and wickets, the other 2 being cumulative runs and wickets in past `4` overs, which introduces the factor of long term and short term scores and wickets.

Then I converted `batting_team` and `bowling_team` to hot-encoded format for better representation.

Furthur I added another column that contains prediction from run_rates, i.e. considering how cumulative runs changes till the overs provided, from visualisation it always seems a linear curve, so a linear fit from numpy gave a prediction score at end of 20 overs (considering 6 balls per over).

All these columns are now passes in a linear_regression object, whose predictions are inserted into `predictions.csv`.
The printed lines in `std::out` like:

```bash
Match ID: 304 , Pred_Model: 166 , Pred_extrapolation: 134.09 , Final_pred: 159.61800000000002
```

Pred_extrapolation is the prediction from cumulative run rate.
