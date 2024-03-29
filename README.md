# Full Takeoff Model
Work in progress! I don't recommend looking at the code right now. It needs to be cleaned up. 

For now, you might be interested in [the core file](ftm/core/model.py).

Instructions to execute the model:
- Clone the github repo 
- Set up the environment (this project uses [Poetry](https://python-poetry.org/docs/)) with `poetry install`
- Create a copy of the [Task_based_model_inputs](https://docs.google.com/spreadsheets/d/1OMBcEvWgCVut305CRBPxLtxjxVP4m_DOiMu9qDUgHso/) and set the parameters as you want, and make it public
- Run `poetry run python -m ftm.core.model -w YOUR_SHEET_URL` to run the model with best_guess parameters
- Run `poetry run python -m ftm.analysis.exploration_analysis -w YOUR_SHEET_URL` to run a comparison of the model with aggressive, best_guess and conservative parameter choices
    - You can run `poetry run python -m ftm.analysis.exploration_analysis -w YOUR_SHEET_URL -t PARAMETER_NAME` instead to produce a detailed comparison of what happens when you set said parameter at aggressive, best_guess and conservative, with all other parameters held at best_guess value
- Run `poetry run python -m ftm.analysis.sensitivity_analysis -w YOUR_SHEET_URL` to see a high level comparison of what happens when you change each parameter between their aggressive, best_guess and conservative values, having all other parameters fixed at their best_guess value
    - The parameters will appear ordered by most sensitive to least sensitive. Concretely, the difference between the takeoff_length of the conservative and aggressive value, for a complex definition of takeoff_length
- Run `poetry run python -m ftm.analysis.mc_analysis -w YOUR_SHEET_URL` to run a MC sampling
    - The aggressive, best guess and conservative values correspond to percentiles 5%, 50% and 95% of a distribution
    - You can adjust the correlation between parameters using the rank_correlation_between_buckets tab in the sheet
- Run `poetry run python -m ftm.analysis.timelines_report -w YOUR_SHEET_URL` to run nine scenarios corresponding to the conservative, best_guess and aggresive choices, conditioned on short, best_guess and long AI timelines 
    - The full_automation_requirement and flop_gap parameters are governed by the sheet Guess FLOP gap and timelines
- Run `poetry run python -m ftm.analysis.megareport -w YOUR_SHEET_URL` to run the three previous analysis at once
