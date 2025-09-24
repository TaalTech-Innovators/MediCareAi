def drop_unwanted_columns(df):
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]