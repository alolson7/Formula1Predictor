# script for data preprocessing

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os
import glob


# downloads most recent data from Vopani's kaggle dataset to ./data/
# NOTE: if you get the error "OSError: Could not find kaggle.json", add the kaggle.json file to
# C:Users/(computer username)/.kaggle/ after doing pip install kaggle
def get_kaggle_data():
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('rohanrao/formula-1-world-championship-1950-2020', path="./data/")

    # unzip
    with zipfile.ZipFile('./data/formula-1-world-championship-1950-2020.zip', 'r') as zipref:
        zipref.extractall('./data/')

    # delete zip file
    os.remove('./data/formula-1-world-championship-1950-2020.zip')


# clear ./data/
def clear_data():
    files = glob.glob('./data/*')

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("clear_data error:", f, ":", e.strerror)


def preprocess():
    results = pd.read_csv(r'data\results.csv')
    races = pd.read_csv(r'data\races.csv')
    quali = pd.read_csv(r'data\qualifying.csv')
    drivers = pd.read_csv(r'data\drivers.csv')
    constructors = pd.read_csv(r'data\constructors.csv')
    circuit = pd.read_csv(r'data\circuits.csv')

    df1 = pd.merge(races, results, how='inner', on=['raceId'])
    df2 = pd.merge(df1, quali, how='inner', on=['raceId', 'driverId', 'constructorId'])
    df3 = pd.merge(df2, drivers, how='inner', on=['driverId'])
    df4 = pd.merge(df3, constructors, how='inner', on=['constructorId'])
    df5 = pd.merge(df4, circuit, how='inner', on=['circuitId'])

    # drop the columns which are not important
    data = df5.drop(['round', 'circuitId', 'time_x', 'url_x', 'resultId', 'driverId',
                     'constructorId', 'number_x', 'positionText', 'position_x',
                     'positionOrder', 'laps', 'time_y', 'rank',
                     'fastestLapTime', 'fastestLapSpeed', 'qualifyId', 'driverRef', 'number', 'code', 'url_y',
                     'circuitRef',
                     'location', 'lat', 'lng', 'alt', 'number_y', 'points', 'constructorRef', 'name_x', 'raceId',
                     'fastestLap', 'q2', 'q3', 'milliseconds', 'q1'], 1)

    # considering data points from 2010
    data = data[data['year'] >= 2010]

    # rename the columns
    data.rename(columns={'name': 'GP_name', 'position_y': 'position', 'grid': 'quali_pos', 'name_y': 'constructor',
                         'nationality_x': 'driver_nationality', 'nationality_y': 'constructor_nationality'},
                inplace=True)
    data['driver'] = data['forename'] + ' ' + data['surname']
    data['date'] = pd.to_datetime(data['date'])
    data['dob'] = pd.to_datetime(data['dob'])

    # creating a driver age parameter
    data['age_at_gp_in_days'] = abs(data['dob'] - data['date'])
    data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: str(x).split(' ')[0])

    # Some of the constructors changed their name over the year so replacing old names with current name
    data['constructor'] = data['constructor'].apply(lambda x: 'Racing Point' if x == 'Force India' else x)
    data['constructor'] = data['constructor'].apply(lambda x: 'Alfa Romeo' if x == 'Sauber' else x)
    data['constructor'] = data['constructor'].apply(lambda x: 'Renault' if x == 'Lotus F1' else x)
    data['constructor'] = data['constructor'].apply(lambda x: 'AlphaTauri' if x == 'Toro Rosso' else x)

    data['driver_nationality'] = data['driver_nationality'].apply(lambda x: str(x)[:3])
    data['constructor_nationality'] = data['constructor_nationality'].apply(lambda x: str(x)[:3])
    data['country'] = data['country'].apply(lambda x: 'Bri' if x == 'UK' else x)
    data['country'] = data['country'].apply(lambda x: 'Ame' if x == 'USA' else x)
    data['country'] = data['country'].apply(lambda x: 'Fre' if x == 'Fra' else x)
    data['country'] = data['country'].apply(lambda x: str(x)[:3])
    data['driver_home'] = data['driver_nationality'] == data['country']
    data['constructor_home'] = data['constructor_nationality'] == data['country']
    data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
    data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

    # reasons for DNF(did not finish)
    data['driver_dnf'] = data['statusId'].apply(
        lambda x: 1 if x in [3, 4, 20, 29, 31, 41, 68, 73, 81, 97, 82, 104, 107, 130, 137] else 0)
    data['constructor_dnf'] = data['statusId'].apply(
        lambda x: 1 if x not in [3, 4, 20, 29, 31, 41, 68, 73, 81, 97, 82, 104, 107, 130, 137, 1] else 0)
    data.drop(['forename', 'surname'], 1, inplace=True)


    dnf_by_driver = data.groupby('driver').sum()['driver_dnf']
    driver_race_entered = data.groupby('driver').count()['driver_dnf']
    driver_dnf_ratio = (dnf_by_driver / driver_race_entered)
    driver_confidence = 1 - driver_dnf_ratio
    driver_confidence_dict = dict(zip(driver_confidence.index, driver_confidence))

    dnf_by_constructor = data.groupby('constructor').sum()['constructor_dnf']
    constructor_race_entered = data.groupby('constructor').count()['constructor_dnf']
    constructor_dnf_ratio = (dnf_by_constructor / constructor_race_entered)
    constructor_relaiblity = 1 - constructor_dnf_ratio
    constructor_relaiblity_dict = dict(zip(constructor_relaiblity.index, constructor_relaiblity))

    data['driver_confidence'] = data['driver'].apply(lambda x: driver_confidence_dict[x])
    data['constructor_reliability'] = data['constructor'].apply(lambda x: constructor_relaiblity_dict[x])

    # removing retired drivers and constructors
    active_constructors = ['Renault', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                           'AlphaTauri', 'Racing Point', 'Alfa Romeo', 'Red Bull',
                           'Haas F1 Team']
    active_drivers = ['Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                      'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                      'Lando Norris', 'Sebastian Vettel', 'Kimi Räikkönen',
                      'Charles Leclerc', 'Lewis Hamilton', 'Daniil Kvyat',
                      'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                      'Sergio Pérez', 'Esteban Ocon', 'Antonio Giovinazzi',
                      'Romain Grosjean', 'Nicholas Latifi']
    data['active_driver'] = data['driver'].apply(lambda x: int(x in active_drivers))
    data['active_constructor'] = data['constructor'].apply(lambda x: int(x in active_constructors))
    
    return data


get_kaggle_data()
# clear_data()