import wandb
import pandas as pd

# Replace with your W&B username and project name
api = wandb.Api()
runs = api.runs("jashing223-national-taiwan-normal-university/pkws")

# Collect run info into a list of dicts
summary_list = []
for run in runs:
    history = run.history()
    history.to_csv(f"./record/{run.name}_history.csv", index=False)
    # Convert to DataFrame
    df = pd.DataFrame(summary_list)