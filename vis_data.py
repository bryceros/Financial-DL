import pandas as pd

# def main():
#     pass


def process():
     data = pd.read_csv('data/permno_data_ticker_comnam_permco_prc_vol_bid_ask')
     data['date'] = pd.to_datetime(data['date'])
     c = data[data['date'] >= pd.datetime(2008, 1, 1)]
     c = c[c.TICKER != "SYF"]
     c = c[c.TICKER != "MLSS"]
     c = c[c.TICKER != "ESV"]
     c.to_csv('data/pre_data_10years')

process()