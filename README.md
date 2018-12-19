# How to run code for noise generation dataset
python main.py folder_name dataset_xlsx_file percentage_of_noise classes_or_target_classes

# Example for target feature noise only
python main.py nba_players player_stats.xslx 0.05 c 
This will produce xlsx with the format player_stats_0.05_c.xlsx inside folder nba_players

# Example for features noise only
python main.py nba_players player_stats.xslx 0.05 f
This will produce xlsx with the format player_stats_0.05_f.xlsx inside folder nba_players


run for t-test
python air.py AirQualityUCI_0.35_c_train_noisy.xlsx  AirQualityUCI_0.35__test_.xlsx


