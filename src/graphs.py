from pathlib import Path
import matplotlib
import pandas as pd

# Read CSV into DataFrame
df = pd.read_csv(Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv')

# Print first few rows
print(df.head())
