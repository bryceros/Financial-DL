import pandas as pd
import numpy as np

def parse(file_name):
    data = pd.read_csv(file_name)
    dates = data['date'].unique()
    companies = list(data['TICKER'].unique())
    
    companies.remove(np.nan)
    companies.remove('LNX')

    attributes =  ['PRC','VOL','BID','ASK']

    columns = ['date']
    for c in companies:
        if isinstance(c, str):
            for a in attributes:
                columns.append(c+'_'+a)

    df = pd.DataFrame(columns=columns)
    for date in dates:
        data_date = data[data['date'] == date]
        if data_date.shape[0] >= len(companies)-1:
            row = {'date':date}
            for c in companies:
                if isinstance(c, str):
                    for a in attributes:
                        s = data_date[data_date['TICKER'] == c]
                        s1 = s[a]
                        row[c+'_'+a] = data_date[data_date['TICKER'] == c][a].values[0]
            df = df.append(row, ignore_index=True)
    #df.to_csv('data/db.csv')

    return df
