###########################################################################
#                    AI in Production Engineering                         #
#                             SS 2023                                     #
#                                                                         #
#                 Predictive Maintenance Group 5                          #
#                                                                         #
#                                                                         #
#                                                                         #
#                                                                         #
###########################################################################
def preprocessing(data):
    # TODO: this implementation is only suited to process train_FD001
    df=data.drop([26,27], axis=1)
    df.name = "train_FD001"
    # define header
    header_txt = ["unit_number","time_cycle", "setting_1", "setting_2", "setting_3",
                "sensor_1", "sensor_2", "sensor_3","sensor_4","sensor_5","sensor_6", "sensor_7", "sensor_8", "sensor_9", 
                "sensor_10", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", 
                "sensor_18", "sensor_19", "sensor_20", "sensor_21"] 
    df.columns = header_txt # add header to datafile

    remaining_names = ['sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12']
    sensor_names = ['sensor_{}'.format(i+1) for i in range(0,21)]
    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in remaining_names]
    X_train_pre = df.drop(drop_sensors, axis=1)

    #########################################
    # TODO: add reference RUL to datapoints


    # TODO: include Health indicator

    # TODO: inlcude normalization

    #########################################
    return X_train_pre
