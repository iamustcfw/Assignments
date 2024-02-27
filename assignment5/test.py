import pandas as pd

def load_data(path):
    df = pd.read_json(path)
    df_expanded = df['user'].apply(lambda x: pd.Series(x))
    df = pd.concat([df.drop('user', axis=1), df_expanded], axis=1)
    return df

data_train = load_data("./data/train.json") 

# 获取 'utc_offset' 字段的所有独特值
utc_offset_unique = data_train['utc_offset'].unique()
print("Unique values in 'utc_offset':", utc_offset_unique)

# 获取 'time_zone' 字段的所有独特值
time_zone_unique = data_train['time_zone'].unique()
print("Unique values in 'time_zone':", time_zone_unique)
