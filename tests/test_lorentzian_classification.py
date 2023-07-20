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
    lc.dump('output/result.csv')

    df1 = pd.read_csv('output/result.csv')
    df2 = pd.read_csv('expected/lc_case1.csv')

    assert df1.equals(df2)
