# How to run code for noise generation dataset
python main.py folder_name dataset_xlsx_file percentage_of_noise classes_or_target_classes

# Example for target feature noise only
python main.py nba_players player_stats.xslx 0.05 c 

# Example for features noise only
python main.py nba_players player_stats.xslx 0.05 f

This will produce xlsx with the format player_stats_0.05_c.xlsx
