from pathlib import Path
import matplotlib
import csv

r = csv.reader((Path(__file__).parent / '..' / 'pipeline' / '4_analysis' / 'trial_results_aggregated.csv').read_text().splitlines())

print(next(r))
