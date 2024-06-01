def data_validation(data, features_metadata):
    user_feature_values, warnings = [], []
    for feature, feature_metadata in features_metadata.items():
        user_value = data.get(feature)
        if user_value is None or user_value == '':
            raise ValueError(f'Missing value for {feature}')
        
        if feature_metadata['dtype'] == 'boolean':
            if isinstance(user_value, str):
                if user_value.lower() == 'true':
                    user_value = True
                elif user_value.lower() == 'false':
                    user_value = False
                else:
                    raise ValueError(f'Invalid value for {feature}: {user_value}')
            elif not isinstance(user_value, bool):
                raise ValueError(f'Invalid value for {feature}: {user_value}')
            user_feature_values.append(user_value)
        
        elif feature_metadata['dtype'] == 'string':
            if user_value in feature_metadata['unique_values']:
                user_feature_values.append(user_value)
            else:
                raise ValueError(f'Invalid value for {feature}: {user_value}')
        
        elif feature_metadata['dtype'] == 'numerical':
            user_value = float(user_value)  # Ensure the value is a float
            user_feature_values.append(user_value)
            if user_value < feature_metadata['min'] or user_value > feature_metadata['max']:
                warnings.append(f'Warning: the inserted value for \'{feature}\' ({user_value}) is out of the training range for this variable.')
    
    return user_feature_values, warnings