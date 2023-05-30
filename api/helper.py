
column_processing=[
        {
            "column_name1":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"0-1"
            },
            "column_name2":{
                "missing_value":"average",
                "encoding":"one_hot",
                "feature_scaling":"-1-1"
            }
        }
    ]

def process_df(df,column_processing,x_cols):
    for col in x_cols:
        missing_value = column_processing[col+'_missing_value']
        if missing_value=='forward_fill':
            df[col] = df[col].ffill()
        elif missing_value=='backward_fill':
            df[col] = df[col].bfill()
        elif missing_value=='interpolate':
            df[col] = df[col].interpolate()

        encoding = column_processing[col+'_encoding']


    return df
