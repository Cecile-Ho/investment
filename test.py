import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 定義 ETF 類別
# ------------------------------
class ETF:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
    
    def download_data(self, start="2015-01-01", end="2025-12-01"):
        """
        下載 ETF 調整後收盤價 (Adj Close)，避免 MultiIndex 問題
        """
        raw_data = yf.download(self.ticker, start=start, end=end, group_by='ticker', auto_adjust=False)
        
        # 處理多支 ETF 或單支 ETF 的差異
        if isinstance(raw_data.columns, pd.MultiIndex):
            # 多支 ETF 或包含 OHLCV MultiIndex
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                self.data = raw_data['Adj Close']
            else:
                # 只有一支 ETF，取第一層欄位
                self.data = raw_data[self.ticker]['Adj Close']
        else:
            # 單層欄位，直接取 Adj Close
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            else:
                # 沒有 Adj Close，直接用 Close
                self.data = raw_data['Close']

        self.data = self.data.fillna(method='ffill')
        return self.data

# ------------------------------
# 定義策略基類
# ------------------------------
class Strategy:
    def generate_signals(self, portfolio):
        """
        輸入: Portfolio 物件
        輸出: dict {ticker: signal}，signal: 1=買, -1=賣, 0=無操作
        """
        raise NotImplementedError("策略需實作 generate_signals 方法")

# ------------------------------
# 定義定期定額策略
# ------------------------------
class DollarCostAveraging(Strategy):
    def generate_signals(self, portfolio):
        # 每月都買入
        signals = {etf.ticker: 1 for etf in portfolio.etfs}
        return signals

# ------------------------------
# 定義投資組合
# ------------------------------
class Portfolio:
    def __init__(self, etfs, allocation, initial_cash=1000):
        """
        etfs: ETF 物件列表
        allocation: np.array 與 etfs 對應的資產比例
        """
        self.etfs = etfs
        self.allocation = allocation
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {etf.ticker: 0 for etf in etfs}
        self.portfolio_value = []
        self.dates = None
    
    def backtest(self, strategy, monthly_investment=100):
        # 假設所有 ETF 的資料日期一致
        self.dates = self.etfs[0].data.index
        for i, date in enumerate(self.dates):
            # 每月交易
            if i % 21 == 0:
                signals = strategy.generate_signals(self)
                prices = np.array([etf.data.iloc[i] for etf in self.etfs])
                invest_amounts = monthly_investment * self.allocation
                for j, etf in enumerate(self.etfs):
                    if signals[etf.ticker] == 1:
                        shares_to_buy = invest_amounts[j] // prices[j]
                        self.holdings[etf.ticker] += shares_to_buy
                        self.cash -= shares_to_buy * prices[j]
            
            # 計算當前組合價值
            total_value = self.cash + sum(
                self.holdings[etf.ticker] * etf.data.iloc[i] for etf in self.etfs
            )
            self.portfolio_value.append(total_value)
        
        return pd.Series(self.portfolio_value, index=self.dates)
    
    def plot_portfolio(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.dates, self.portfolio_value, label='Portfolio Value')
        plt.title('Portfolio Backtest')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# 建立 ETF 物件
stock_etf = ETF('SPY')
bond_etf = ETF('AGG')

# 下載歷史資料
stock_etf.download_data()
bond_etf.download_data()

# 建立投資組合（股票 60%，債券 40%）
portfolio = Portfolio([stock_etf, bond_etf], allocation=np.array([0.6, 0.4]))

# 定義策略
strategy = DollarCostAveraging()

# 回測
portfolio.backtest(strategy, monthly_investment=5000)

# 畫出組合曲線
portfolio.plot_portfolio()
