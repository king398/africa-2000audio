import pandas as pd
df_0 = pd.read_csv("oof/predictions_train_0.csv")
df_1 = pd.read_csv("oof/predictions_train_1.csv")
df = pd.concat([df_0, df_1]).reset_index(drop=True)
df['predictions'] = df['predictions'].str.replace('"', '', regex=True)
df['labels'] = df['labels'].str.replace('"', '', regex=True)
df.to_csv("oof/predictions_train.csv", index=False)
