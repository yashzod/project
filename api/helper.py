
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

def process_df(df,column_processing):
    columns = df.columns
    return df
