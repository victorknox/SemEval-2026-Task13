import pandas as pd

for task in ['A', 'B', 'C']:
    df = pd.read_parquet(f'task_{task}/task_{task.lower()}_trial.parquet')
    print(f'\n=== TASK {task} ===')
    print(f'Samples: {len(df)}')
    print(f'Unique labels: {df["label"].nunique()}')
    print(f'Label distribution:')
    print(df["label"].value_counts().sort_index())
