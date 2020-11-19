"""
mazzi888's script was instrumental for pulling data from The Graph and thinking about how 
to approach this analysis: https://github.com/mazzi888/HegicGreeks/blob/master/greeks.py
"""

import requests
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
import py_vollib.black_scholes.greeks.numerical as gn
import time
import numpy as np

class GetData:
    def __init__(self, query, asset='Ethereum'):
        """
        Initializing `GetData` class to pull all relevant Hegic data.

        Parameters:
        ----------
        query: str
            The GraphQL Query commands used to pull data from the Graph Protocol
        asset: str
            The options pool we want to analyze - choose between "Bitcoin" or "Ethereum" (default "Ethereum")
        """
        self.query = query
        self.asset = asset.lower()
        if self.asset =='bitcoin':
            self.ticker = 'WBTC'
        else:
            self.ticker = 'ETH'
        spot_url = "https://api.coingecko.com/api/v3/simple/price?ids=" + self.asset + "&vs_currencies=usd"
        self.spot = requests.get(spot_url).json()[self.asset]['usd']

    def run_query(self):
        """
        Downloads the raw data from the Hegic V888 subgraph and converts the output into a dataframe
        """
        request = requests.post('https://api.thegraph.com/subgraphs/name/ppunky/hegic-v888', json={'query': self.query})
        if request.status_code != 200:
            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, self.query))

        data = request.json()['data']['options']
        df = pd.DataFrame(data)
        return df

    def format_data(self):
        """
        Formats the dataframe to ensure the following criteria:
            1. Status of the option must be active
            2. Time to expiry must be greater than 0
            3. Strike of the option must be greater than 0

        ***KEY ASSUMPTION***
        --------------------------------------------------------------------------------
        This method will also calculate the current USD price of the *original* ETH or BTC premium paid
        by the option buyer. Therefore, the current USD price of the original option premium will be used
        as a rough proxy for calculating the implied volatility and greeks of the option. This approach is
        flawed as it does not accurately reflect the current value of the option. Once a better method is found
        this can be adjusted accordingly.
        """
        df = self.run_query()
        # Convert the time to expiry into years
        df['time_to_expiry'] = ((df.expiration - time.time())/(60*60*24))/(365)
        # Convert these columns into numeric values
        numeric_columns = ['premium', 'amount', 'strike']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        # Filter out the options based on the criteria listed above
        df = df[(df.symbol==self.ticker) & (df.status=="ACTIVE") & \
                (df.time_to_expiry > 0) & (df.strike > 0)].reset_index().drop('index', axis=1)
        # Rename the option types to be compatiable with the `py_vollib` library
        df['type'] = np.where(df['type']=='CALL', 'c', 'p')
        # Calculate the current USD price of the premium when it was originally bought
        df['premium_usd'] = (df.premium*self.spot/df.amount)
        return df

    def get_iv(self):
        """
        Calculates the implied volatility for each individual option by using the following
        input values (interest rates assumed to be zero):
            1. Spot Price of ETH or BTC
            2. Time to Maturity (years)
            3. Strike Price
            4. USD Option Premium (as calculated in `format_data`)

        Note: The IV approximation function fails to converge to a value for several options 
        (around 10-15% of total option volume).
        """
        iv_store = []
        df = self.format_data()
        for option in range(len(df)):
            try:
                iv_store.append(iv(price=df['premium_usd'].iloc[option],
                                      S = self.spot,
                                      K = df['strike'].iloc[option],
                                      t = (df['time_to_expiry'].iloc[option]),
                                      r = 0,
                                      flag = df['type'].iloc[option]))
            except:
                # In the case we can't solve IV we store a NaN value
                iv_store.append(np.nan)
                pass

        return iv_store

    def get_greeks(self):
        """
        Calculates the individual option greeks for each Hegic option. Once all the greeks
        are calculated, they are multiplied by -1 to reflect the short position of all LPs.

        Note: Only options with non-NaN implied volatilities will have greek values (recall
        from `get_iv` that 10-15% of the total volume will encounter NaN values.
        """
        df = self.format_data()
        df['iv'] = self.get_iv()
        greek_store = []
        # Calculate the greeks for each individual option
        for option in range(len(df)):
            try:
                delta = gn.delta(S=self.spot,
                                 K=df.strike[option],
                                 t=df.time_to_expiry[option],
                                 r=0,
                                 sigma=df.iv[option],
                                 flag=df['type'][option])

                gamma = gn.gamma(S=self.spot,
                                 K=df.strike[option],
                                 t=df.time_to_expiry[option],
                                 r=0,
                                 sigma=df.iv[option],
                                 flag=df['type'][option])

                theta = gn.theta(S=self.spot,
                                 K=df.strike[option],
                                 t=df.time_to_expiry[option],
                                 r=0,
                                 sigma=df.iv[option],
                                 flag=df['type'][option])

                vega = gn.vega(S=self.spot,
                                 K=df.strike[option],
                                 t=df.time_to_expiry[option],
                                 r=0,
                                 sigma=df.iv[option],
                                 flag=df['type'][option])

                greek_store.append([delta, gamma, theta, vega])
            except:
                greek_store.append([np.nan, np.nan, np.nan, np.nan])
        # Multiply the greeks by -1 to account for the short LP position
        greeks = pd.DataFrame(greek_store)*-1
        greeks.columns = ['delta', 'gamma', 'theta', 'vega']
        df = pd.concat([df, greeks], axis=1)
        return df
