# opmodel

Instructions to execute the model
- Clone the github repo 
- Create a copy of the (Task_based_model_inputs)[https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI] and set the parameters as you want
- Run `python -m opmodel.core.opmodel -w YOUR_SHEET_URL` to run the model with best_guess parameters
- Run `python -m opmodel.analysis.exploration_analysis -w YOUR_SHEET_URL` to run a comparison of the model with aggressive, best_guess and conservative parameter choices
    - You can run python -m opmodel.analysis.exploration_analysis -w YOUR_SHEET_URL -t PARAMETER_NAME instead to produce a detailed comparison of what happens when you set said parameter at aggressive, best_guess and conservative, with all other parameters held at best_guess value
- Run `python -m opmodel.analysis.sensitivity_analysis -w YOUR_SHEET_URL` to see a high level comparison of what happens when you change each parameter between their aggressive, best_guess and conservative values, having all other parameters fixed at their best_guess value
    - The parameters will appear ordered by most sensitive to least sensitive. Concretely, the difference between the takeoff_length of the conservative and aggressive value, for a complex definition of takeoff_length
- Run `python -m opmodel.analysis.mc_analysis -w YOUR_SHEET_URL` to run a MC sampling
    - The aggressive, best guess and conservative values correspond to percentiles 5%, 50% and 95% of a distribution
    - You can adjust the correlation between parameters using the rank_correlation_between_buckets tab in the sheet
- Run `python -m opmodel.analysis.timelines_report -w YOUR_SHEET_URL` to run nine scenarios corresponding to the conservative, best_guess and aggresive choices, conditioned on short, best_guess and long AI timelines 
    - The full_automation_requirement and flop_gap parameters are governed by the sheet Guess FLOP gap and timelines
- Run `python -m opmodel.analysis.megareport -w YOUR_SHEET_URL` to run the three previous analysis at once
