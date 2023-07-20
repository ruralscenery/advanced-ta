import pytest
import os
import shutil
import pandas as pd
from advanced_ta import LorentzianClassification

os.chdir('tests')


@pytest.fixture
def setup():
    try: os.mkdir('output')
    except FileExistsError: pass
    

def teardown():
    shutil.rmtree(os.path.join(os.getcwd(), 'output'), ignore_errors=True)


def load_data(file: str):
    df = pd.read_csv(file)
    df['date'] = pd.DatetimeIndex(df['date'])
    df.set_index("date", inplace=True)
    return df


def test_case1(setup):
    df = load_data("data/NSE_Instruments_256265.csv")

    lc = LorentzianClassification(df)
    lc.dump('output/result1.csv')

    df1 = pd.read_csv('output/result1.csv')
    df2 = pd.read_csv('expected/lc_case1.csv')

    assert df1.equals(df2)


def test_case2(setup):
    df = load_data("data/NSE_Instruments_256265.csv")

    lc = LorentzianClassification(
        df,
        features=[
            LorentzianClassification.Feature("RSI", 14, 2),  # f1
            LorentzianClassification.Feature("WT", 10, 11),  # f2
            LorentzianClassification.Feature("CCI", 20, 2),  # f3
            LorentzianClassification.Feature("ADX", 20, 2),  # f4
            LorentzianClassification.Feature("RSI", 9, 2),   # f5
        ],
        settings=LorentzianClassification.Settings(
            source='close',
            neighborsCount=8,
            maxBarsBack=2000,
            showDefaultExits=False,
            useDynamicExits=False
        ),
        filterSettings=LorentzianClassification.FilterSettings(
            useVolatilityFilter=True,
            useRegimeFilter=True,
            useAdxFilter=False,
            regimeThreshold=-0.1,
            adxThreshold=20
        ))
    lc.dump('output/result2.csv')

    df1 = pd.read_csv('output/result2.csv')
    df2 = pd.read_csv('expected/lc_case1.csv')

    assert df1.equals(df2)
