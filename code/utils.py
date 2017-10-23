import datetime
import pickle

filesafe_replacements = str.maketrans(" :", "_-")

def datetime_for_filename():
    return str(datetime.datetime.now()).translate(filesafe_replacements)

def load_from_pickle_if_possible(pkl, alt_load_fn):
    try:
        print('Trying to load from pickle...')
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        print('Loaded from pickle.')
    except FileNotFoundError as e:
        print('Pickle not found. Loading original images...')
        data = alt_load_fn()
        with open(pkl, 'wb') as f:
            pickle.dump(data, f)
        print('Saved as pickle.')
    return data
