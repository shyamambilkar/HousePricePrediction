import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("Load saved artifacts....Start...!!")
    global __data_columns
    global __locations

    with open("./artifacts/columns1.json", 'r') as file:
        __data_columns = json.load(file)['data_columns']
        __locations = __data_columns[3:]  # First three columns total_sqft", "bath", "bhk"

    global __model

    if __model is None:
        with open("./artifacts/banglore_home_prices_model1.pickle", 'rb') as file:
            __model = pickle.load(file)
        print("Loading Saved artifacts....done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar',1000,2,2))
