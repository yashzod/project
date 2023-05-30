import pandas as pd
import hashlib


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
        unique_count = df[col].nunique()
        if unique_count < 10:
            if encoding == 'One Hot':
                def one_hot_encode(df, column_name):
                    # Perform one-hot encoding using pandas get_dummies function
                    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
                    
                    # Concatenate the one-hot encoded columns to the original DataFrame
                    df = pd.concat([df, one_hot_encoded], axis=1)
                    
                    # Drop the original column from the DataFrame
                    df = df.drop(column_name, axis=1)
                    
                    return df
                df = one_hot_encode(df,col)
            if encoding == 'Effect' :
                

                def effect_encode(df, column_name):
                    # Create a copy of the original DataFrame
                    encoded_df = df.copy()
                    
                    # Get the unique levels of the column
                    levels = df[column_name].unique()
                    
                    # Define the reference level (e.g., the first level)
                    reference_level = levels[0]
                    
                    # Iterate over the levels and encode each level
                    for level in levels:
                        if level == reference_level:
                            # Skip encoding for the reference level
                            continue
                            
                        # Create a new column with the effect encoding for the level
                        encoded_df[f'{column_name}_{level}_effect'] = (df[column_name] == level).astype(int) - (df[column_name] == reference_level).astype(int)
                    
                    # Drop the original column from the DataFrame
                    encoded_df = encoded_df.drop(column_name, axis=1)
                    
                    return encoded_df
                effect_encode(df, col)

            if encoding == 'Binary':
                def binary_encode(df, column_name):
                    # Create a copy of the original DataFrame
                    encoded_df = df.copy()
                    
                    # Get the unique levels of the column
                    levels = df[column_name].unique()
                    
                    # Determine the number of bits needed to represent the maximum value
                    num_bits = len(bin(len(levels)-1)[2:])
                    
                    # Create a binary code for each level
                    for level in levels:
                        binary_code = bin(levels.tolist().index(level))[2:].zfill(num_bits)
                        for i, bit in enumerate(binary_code):
                            encoded_df[f'{column_name}_bit{i+1}'] = (df[column_name] == level).astype(int) if bit == '1' else 0
                    
                    # Drop the original column from the DataFrame
                    encoded_df = encoded_df.drop(column_name, axis=1)
                    
                    return encoded_df
                binary_encode(df, col)
            if encoding == 'BaseN':
            
                def base_n_encode(df, column_name, base):
                # Create a copy of the original DataFrame
                    encoded_df = df.copy()

                    # Get the unique levels of the column
                    levels = df[column_name].unique()

                    # Create a base-N code for each level
                    for level in levels:
                        base_n_code = base_encode(levels.tolist().index(level), base)
                        for i, digit in enumerate(base_n_code):
                            encoded_df[f'{column_name}_digit{i+1}'] = (df[column_name] == level).astype(int) if digit == '1' else 0

                    # Drop the original column from the DataFrame
                    encoded_df = encoded_df.drop(column_name, axis=1)

                    return encoded_df

                def base_encode(number, base):
                    # Function to encode a number in the specified base
                    encoding = ''
                    while number > 0:
                        encoding = str(number % base) + encoding
                        number = number // base
                    return encoding
                base_n_encode(df, col, base=3)

            if encoding == 'Hash':
                def hash_encode(df, column_name, hash_type='md5'):
                    # Create a copy of the original DataFrame
                    encoded_df = df.copy()
                    
                    # Get the unique levels of the column
                    levels = df[column_name].unique()
                    
                    # Hash encode each level
                    for level in levels:
                        hash_value = hash_string(level, hash_type)
                        encoded_df[f'{column_name}_hash'] = (df[column_name] == level).astype(int) if hash_value == level else 0
                    
                    # Drop the original column from the DataFrame
                    encoded_df = encoded_df.drop(column_name, axis=1)
                    
                    return encoded_df

                def hash_string(value, hash_type):
                    # Function to hash a string using the specified hash type
                    hasher = hashlib.new(hash_type)
                    hasher.update(value.encode('utf-8'))
                    return hasher.hexdigest()

                hash_encode(df, col, hash_type='md5')

            # if encoding == 'Target':
            #     def target_encode(df, column_name, target_column):
            #         # Create a copy of the original DataFrame
            #         encoded_df = df.copy()

            #         # Calculate the mean target value for each unique level in the column
            #         mean_target = df.groupby(column_name)[target_column].mean()

            #         # Map the mean target values to the corresponding levels in the column
            #         encoded_df[f'{column_name}_target_encoded'] = df[column_name].map(mean_target)

            #         return encoded_df
            #     target_encode(df, col, 'Target')
        else:
            df = df.drop(col, axis=1)



    return df
