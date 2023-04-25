import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
#from utils import Plot_OHCL
import os
os.environ['PYTHONHASHSEED']=str(1000)
#import fundamentalanalysis as fa
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.losses import MeanSquaredError

def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    #df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Moving Average Convergence Divergence (MACD) indicator
    df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas

    # Add Relative Strength Index (RSI) indicator
    df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas
    
    return df

def DropCorrelatedFeatures(df, threshold, plot):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
                if columns[j]:
                    columns[j] = False
                    
    selected_columns = df_drop.columns[columns]

    df_dropped = df_drop[selected_columns]

    if plot:
        # Plot Heatmap Correlation
        fig = plt.figure(figsize=(8,8))
        ax = sns.heatmap(df_dropped.corr(), annot=True, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.show()
    
    return df_dropped

def get_trend_indicators(df, threshold=0.5, plot=False):
    df_trend = df.copy()
    
    # add custom trend indicators
    df_trend["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df_trend["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df_trend["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()

    df_trend = add_trend_ta(df_trend, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_trend, threshold, plot)

def get_volatility_indicators(df, threshold=0.5, plot=False):
    df_volatility = df.copy()
    
    # add custom volatility indicators
    # ...

    df_volatility = add_volatility_ta(df_volatility, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_volatility, threshold, plot)

def get_volume_indicators(df, threshold=0.5, plot=False):
    df_volume = df.copy()
    
    # add custom volume indicators
    # ...

    df_volume = add_volume_ta(df_volume, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_volume, threshold, plot)

def get_momentum_indicators(df, threshold=0.5, plot=False):
    df_momentum = df.copy()
    
    # add custom momentum indicators
    # ...

    df_momentum = add_momentum_ta(df_momentum, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_momentum, threshold, plot)

def get_others_indicators(df, threshold=0.5, plot=False):
    df_others = df.copy()
    
    # add custom indicators
    # ...

    df_others = add_others_ta(df_others, close="Close")

    return DropCorrelatedFeatures(df_others, threshold, plot)

def get_all_indicators(df, threshold=0.5, plot=False):
    df_all = df.copy()
    
    # add custom indicators
    # ...

    df_all = add_all_ta_features(df_all, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_all, threshold, plot)

def indicators_dataframe(df, threshold=0.5, plot=False):
    trend       = get_trend_indicators(df, threshold=threshold, plot=plot)
    volatility  = get_volatility_indicators(df, threshold=threshold, plot=plot)
    volume      = get_volume_indicators(df, threshold=threshold, plot=plot)
    momentum    = get_momentum_indicators(df, threshold=threshold, plot=plot)
    others      = get_others_indicators(df, threshold=threshold, plot=plot)
    #all_ind = get_all_indicators(df, threshold=threshold)

    final_df = [df, trend, volatility, volume, momentum, others]
    result = pd.concat(final_df, axis=1)

    return result

def Fundamentals(df, tick=''):
    df['Date'] = pd.to_datetime(df['Date'])
    ticker = tick
    api_key = "2f77d07e7627da55b7936ac2229f54d7"

    df.sort_values('Date', inplace=True)
    # Collect market cap and enterprise value
    entreprise_value = fa.enterprise(ticker, api_key, period="quarter")

    entreprise_value = entreprise_value.T.drop(['symbol','stockPrice','numberOfShares'], axis=1).reset_index()
    entreprise_value['Date'] = pd.to_datetime(entreprise_value['index'])
    entreprise_value = entreprise_value.drop(['index'], axis=1)
    entreprise_value.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, entreprise_value, on='Date')



    # Obtain DCFs over time
    dcf_annually = fa.discounted_cash_flow(ticker, api_key, period="quarter")

    dcf_annually = dcf_annually.T.drop(['Stock Price'], axis=1)
    dcf_annually['Date'] = pd.to_datetime(dcf_annually['date'])
    dcf_annually = dcf_annually.drop(['date'], axis=1)
    dcf_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, dcf_annually, on='Date')

    # Collect the Balance Sheet statements
    balance_sheet_annually = fa.balance_sheet_statement(ticker, api_key, period="quarter")


    balance_sheet_annually = balance_sheet_annually.T.drop(['cik','acceptedDate','reportedCurrency','period','calendarYear','link','finalLink'], axis=1)
    balance_sheet_annually['Date'] = pd.to_datetime(balance_sheet_annually['fillingDate'])
    balance_sheet_annually = balance_sheet_annually.drop(['fillingDate'], axis=1)
    balance_sheet_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, balance_sheet_annually, on='Date')

    # Collect the Income Statements
    income_statement_annually = fa.income_statement(ticker, api_key, period="quarter")

    income_statement_annually = income_statement_annually.T.drop(['cik','acceptedDate','reportedCurrency','period','calendarYear','link','finalLink'], axis=1)
    income_statement_annually['Date'] = pd.to_datetime(income_statement_annually['fillingDate'])
    income_statement_annually = income_statement_annually.drop(['fillingDate'], axis=1)
    income_statement_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, income_statement_annually, on='Date')

    # Collect the Cash Flow Statements
    cash_flow_statement_annually = fa.cash_flow_statement(ticker, api_key, period="quarter")

    cash_flow_statement_annually = cash_flow_statement_annually.T.drop(['cik','acceptedDate','reportedCurrency','period','calendarYear','link','finalLink'], axis=1)
    cash_flow_statement_annually['Date'] = pd.to_datetime(cash_flow_statement_annually['fillingDate'])
    cash_flow_statement_annually = cash_flow_statement_annually.drop(['fillingDate'], axis=1)
    cash_flow_statement_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, cash_flow_statement_annually, on='Date')


    # Show Key Metrics
    key_metrics_annually = fa.key_metrics(ticker, api_key, period="quarter")

    key_metrics_annually = key_metrics_annually.T.drop(['period'], axis=1).reset_index()
    key_metrics_annually['Date'] = pd.to_datetime(key_metrics_annually['index'])
    key_metrics_annually = key_metrics_annually.drop(['index'], axis=1)
    key_metrics_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, key_metrics_annually, on='Date')


    # Show a large set of in-depth ratios
    financial_ratios_annually = fa.financial_ratios(ticker, api_key, period="quarter")


    financial_ratios_annually = financial_ratios_annually.T.drop(['period'], axis=1).reset_index()
    financial_ratios_annually['Date'] = pd.to_datetime(financial_ratios_annually['index'])
    financial_ratios_annually = financial_ratios_annually.drop(['index'], axis=1)
    financial_ratios_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, financial_ratios_annually, on='Date')


    # Show the growth of the company
    growth_annually = fa.financial_statement_growth(ticker, api_key, period="quarter")

    growth_annually = growth_annually.T.drop(['period'], axis=1).reset_index()
    growth_annually['Date'] = pd.to_datetime(growth_annually['index'])
    growth_annually = growth_annually.drop(['index'], axis=1)
    growth_annually.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, growth_annually, on='Date')


    # Download dividend history
    dividends = fa.stock_dividend(ticker, api_key, begin="2018-01-01", end="2020-01-01")
    dividends = dividends.drop(['label','declarationDate','paymentDate','recordDate'], axis=1).reset_index()
    dividends['Date'] = pd.to_datetime(dividends['index'])
    dividends = dividends.drop(['index'], axis=1)
    dividends.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, dividends, on='Date')


    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[:, (df==0).mean() <= 0]

    # Show recommendations of Analysts
    ratings = fa.rating(ticker, api_key)


    ratings_one = pd.get_dummies(ratings.select_dtypes(include=['object']),columns = ratings.select_dtypes(include=['object']).columns)
    ratings_one = ratings_one.reset_index()

    ratings_one['Date'] = pd.to_datetime(ratings_one['date'])
    ratings_one = ratings_one.drop(['date'], axis=1)
    ratings_one.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, ratings_one, on='Date')

    ratings = ratings.drop(['rating','ratingRecommendation','ratingDetailsDCFRecommendation','ratingDetailsROERecommendation','ratingDetailsROARecommendation','ratingDetailsDERecommendation','ratingDetailsPERecommendation','ratingDetailsPBRecommendation'], axis=1).reset_index()
    ratings['Date'] = pd.to_datetime(ratings['date'])
    ratings = ratings.drop(['date'], axis=1)
    ratings.sort_values('Date', inplace=True)
    df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df, ratings, on='Date')
    df['Date'] =  df['Date'].astype(str)
    df =df.set_index('Date')
    df = df.fillna(df.mean())
    df=df.astype('float')
    return df



if __name__ == "__main__":   
    df = pd.read_csv('./BTCUSD_1h.csv')
    df = df.sort_values('Date')
    #df = AddIndicators(df)

    #test_df = df[-400:]

    #Plot_OHCL(df)
    get_others_indicators(df, threshold=0.5, plot=True)