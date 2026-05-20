import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, OptionMenu
from tkinter import font as tkfont
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime
from scipy import stats
from arch import arch_model
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

BG_PAGE    = "#f7f7f5"
BG_WHITE   = "#ffffff"
BORDER     = "#ddd9d0"
TEXT_DARK  = "#1a1a18"
TEXT_MID   = "#5c5a54"
TEXT_MUTED = "#9b9990"

def theme_white(root):
    root.configure(bg=BG_PAGE)
    sty = ttk.Style(root)
    sty.theme_use("clam")
    sty.configure("TFrame",background=BG_PAGE,  relief="flat", borderwidth=0)  
    sty.configure("White.TFrame", background=BG_WHITE, bordercolor=BORDER,lightcolor=BORDER, darkcolor=BORDER, relief="solid", borderwidth=2)  
    sty.configure("FrameTitle.TLabel", background=BG_WHITE, foreground=TEXT_MUTED, font=("Bahnschrift", 9))
    sty.configure("FrameLabel.TLabel",  background=BG_WHITE, foreground=TEXT_MID, font=("Bahnschrift", 10))
    sty.configure("FrameValueSmall.TLabel",  background=BG_WHITE, foreground=TEXT_DARK, font=("Courier New", 10, "bold"))
    sty.configure("FrameValueBig.TLabel", background=BG_WHITE, foreground=TEXT_DARK, font=("Courier New", 20, "bold"))
    sty.configure("TitleDashboard.TLabel", background=BG_PAGE, foreground=TEXT_DARK, font=("Bahnschrift", 25, "bold"))
    sty.configure("TitleDashboardLight.TLabel", background=BG_PAGE, foreground=TEXT_DARK, font=("Bahnschrift", 25))
    sty.configure("TButton", background=TEXT_DARK, foreground=BG_WHITE, bordercolor=TEXT_DARK, lightcolor=TEXT_DARK, darkcolor=TEXT_DARK, font=("Bahnschrift",9,"bold"), padding=(12,5), focusthickness=0, focuscolor=TEXT_DARK)
    sty.map("TButton", background=[("active", TEXT_DARK),("disabled",TEXT_MUTED )], foreground=[("disabled", BG_WHITE)])

class YahooExtract:
    def __init__(self):
        self.extracted = False
        self.output_data = []

    VALID_PERIODS = {"2y","5y","10y","max"}

    def fetch(self, ticker, period_input):
        if not isinstance(ticker, str) or "," in ticker:
            raise ValueError("Only a single ticker string is supported.")
        if period_input not in self.VALID_PERIODS:
            raise ValueError(f"Invalid period '{period_input}'. Valid options: {self.VALID_PERIODS}")

        try:
            raw_df = yf.download(ticker, period=period_input, auto_adjust=True)

            if raw_df.empty:
                return False
            
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            self.output_data = [
                {
                    'date':   row['Date'],
                    'open':   row['Open'],
                    'high':   row['High'],
                    'low':    row['Low'],
                    'close':  row['Close'],
                    'volume': row['Volume'],
                }
                for row in raw_df.reset_index().to_dict(orient='records')
            ]

            self.extracted = True
            self._on_success(ticker)
            return True

        except KeyError as e:
            print(f"Unexpected DataFrame structure: {e}")
            self.extracted = False
            return False
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            self.extracted = False
            return False
        
    def __repr__(self):
        status = f"{len(self.output_data)} rows" if self.extracted else "not extracted"
        return f"YahooExtract({status})"

    def _on_success(self, ticker: str) -> None:
        print(f"Historical data retrieved for {ticker}.")

class RealizedVolDashboard:
    def __init__ (self, root):
        self.root = root
        self.root.title("Realized Volatility Dashboard")
        self.root.geometry("1600x1000")
        self.root.state('zoomed')
        theme_white(root)
        self.yahoo_extract = YahooExtract()
        self.extracted = False
        self.annualization = 250
        self.stock_data = None
        self.return_data = None
        self.volatility_data = None
        self.current_volatility = None
        self.x_vol = None
        self.vol_predict_df = None
        self.q_mod = None
        self.p_mod = None
        self.Rmse = None
        self.setup_ui()
    

    def setup_ui(self):
        # main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)

        title_frame = ttk.Frame(main_frame, padding="10")
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        title_frame.columnconfigure(0, weight=1)
        ttk.Label(title_frame,text="Realized Volatility Analysis and Predictions Dashboard for Indonesian Stocks",style="TitleDashboard.TLabel").grid(row=0, column=0, pady=(5, 0))
        ttk.Label(title_frame, text="by Chris R.S.",style="TitleDashboardLight.TLabel").grid(row=1, column=0, pady=(0, 5))

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.S))
        for col in range(4):
            controls_frame.columnconfigure(col, weight=0)
        f1 = ttk.Frame(controls_frame, padding="4")
        f1.grid(row=0, column=0, sticky=(tk.W, tk.S))
        f2 = ttk.Frame(controls_frame, padding="4")
        f2.grid(row=0, column=1, sticky=(tk.W, tk.S))
        f3 = ttk.Frame(controls_frame, padding="4")
        f3.grid(row=0, column=2, sticky=(tk.W, tk.S))
        f4 = ttk.Frame(controls_frame, padding="4")
        f4.grid(row=0, column=3, sticky=(tk.W, tk.S))

        # Query
        query_frame = ttk.Frame(f1, style="White.TFrame", padding="8")
        query_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 6))
        ttk.Label(query_frame, text="Ticks Query",style="FrameTitle.TLabel").grid(row=0, column=0, columnspan=4,sticky=tk.W, pady=(0, 4))
        ttk.Label(query_frame, text="Search Ticks:",style="FrameLabel.TLabel").grid(row=1, column=0, padx=(0, 2), sticky=tk.W)
        self.tick_var = tk.StringVar(value="BBCA")
        ttk.Entry(query_frame, textvariable=self.tick_var, width=8).grid( row=1, column=1, padx=(0, 8), pady=(2, 4))
        ttk.Label(query_frame, text="Duration:", style="FrameLabel.TLabel").grid(row=1, column=2, padx=(0, 2), pady=(2, 4))
        self.tick_dur = tk.StringVar()
        self.tick_dur.set("2y")
        options = {"max","10y","5y","2y"}
        OptionMenu(query_frame, self.tick_dur, *options).grid(row=1, column=3,padx=(0, 2), sticky=tk.W )
        self.query_button = ttk.Button(query_frame, text="Query", command=self.data_query)
        self.query_button.grid(row=2, column=0, padx=(0, 4), pady=(0, 2), sticky=tk.W)
        self.dequery_button = ttk.Button(query_frame, text="Clear Query",command=self.clear_data, state="disabled")
        self.dequery_button.grid(row=2, column=1, padx=(0, 4), pady=(0, 2), sticky=tk.W)
        self.analyze_button = ttk.Button(query_frame, text="Analyze Returns",command=self.returns_analysis, state="disabled")
        self.analyze_button.grid(row=2, column=2, columnspan=2, pady=(0, 2), sticky=tk.W)

        # Desc
        desc_frame = ttk.Frame(f2, style="White.TFrame", padding="8")
        desc_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        ttk.Label(desc_frame, text="Returns Descriptive", style="FrameTitle.TLabel").grid(row=0, column=0, columnspan=2,sticky=tk.W, pady=(0, 4))
        ttk.Label(desc_frame, text="Average Daily Return:", style="FrameLabel.TLabel").grid(row=1, column=0, padx=(0, 5), sticky=tk.W)
        self.daily_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.daily_return_label.grid(row=1, column=1, padx=(0, 15), sticky=tk.W)
        ttk.Label(desc_frame, text="Annualized Average Daily Return:", style="FrameLabel.TLabel").grid(row=2, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.annual_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.annual_return_label.grid(row=2, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        ttk.Label(desc_frame, text="Current Daily Return:",style="FrameLabel.TLabel").grid(row=3, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.current_daily_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.current_daily_return_label.grid(row=3, column=1, padx=(0, 15), pady=(5, 0), sticky=tk.W)
        ttk.Label(desc_frame, text="Current Annual Daily Return:",style="FrameLabel.TLabel").grid(row=4, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.current_annual_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.current_annual_return_label.grid(row=4, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        ttk.Label(desc_frame, text="Max Return:",style="FrameLabel.TLabel").grid(row=5, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.max_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.max_return_label.grid(row=5, column=1, padx=(0, 15), pady=(5, 0), sticky=tk.W)
        ttk.Label(desc_frame, text="Min Return:",style="FrameLabel.TLabel").grid(row=6, column=0, padx=(0, 5), pady=(5, 0), sticky=tk.W)
        self.min_return_label = ttk.Label(desc_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.min_return_label.grid(row=6, column=1, padx=(0, 5), pady=(5, 0), sticky=tk.W)

        # Vol Analysis
        vol_frame = ttk.Frame(f3, style="White.TFrame", padding="8")
        vol_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(vol_frame, text="Volatility Analysis",style="FrameTitle.TLabel").grid(row=0, column=0, columnspan=4,sticky=tk.W, pady=(0, 4))
        self.vol_button = ttk.Button(vol_frame, text="Analyze Volatility",command=self.vol_analysis, state="disabled")
        self.vol_button.grid(row=1, column=0, columnspan=4, pady=(0, 6), sticky=tk.W)
        ttk.Label(vol_frame, text="Current Daily RV:",style="FrameLabel.TLabel").grid(row=2, column=0, padx=(0, 4), sticky=tk.W)
        self.current_daily_rv = ttk.Label(vol_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.current_daily_rv.grid(row=2, column=1, padx=(0, 12), sticky=tk.W)
        ttk.Label(vol_frame, text="Vol Mean:",style="FrameLabel.TLabel").grid(row=2, column=2, padx=(0, 4), sticky=tk.W)
        self.vol_mean_label = ttk.Label(vol_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.vol_mean_label.grid(row=2, column=3, sticky=tk.W)
        ttk.Label(vol_frame, text="Vol Min:",style="FrameLabel.TLabel").grid(row=3, column=0, padx=(0, 4),pady=(4, 0), sticky=tk.W)
        self.vol_min_label = ttk.Label(vol_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.vol_min_label.grid(row=3, column=1, padx=(0, 12), pady=(4, 0), sticky=tk.W)
        ttk.Label(vol_frame, text="Vol Max:",style="FrameLabel.TLabel").grid(row=3, column=2, padx=(0, 4),pady=(4, 0), sticky=tk.W)
        self.vol_max_label = ttk.Label(vol_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.vol_max_label.grid(row=3, column=3, pady=(4, 0), sticky=tk.W)

        # Regime Analysis
        regime_frame = ttk.Frame(f3, style="White.TFrame", padding="8")
        regime_frame.grid(row=1, column=0, sticky="ew", pady=(0, 0))
        ttk.Label(regime_frame, text="Regime Analysis",style="FrameTitle.TLabel").grid(row=0, column=0, columnspan=2,sticky=tk.W, pady=(0, 4))
        ttk.Label(regime_frame, text="Current Regime:",style="FrameLabel.TLabel").grid(row=1, column=0, padx=(0, 4), sticky=tk.W)
        self.regime_label = ttk.Label(regime_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.regime_label.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(regime_frame, text="Percentile:",style="FrameLabel.TLabel").grid(row=2, column=0, padx=(0, 4),pady=(4, 0), sticky=tk.W)
        self.percentile_label = ttk.Label(regime_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.percentile_label.grid(row=2, column=1, pady=(4, 0), sticky=tk.W)
        ttk.Label(regime_frame, text="MR Forecast:",style="FrameLabel.TLabel").grid(row=3, column=0, padx=(0, 4),pady=(4, 0), sticky=tk.W)
        self.mean_reversion_label = ttk.Label(regime_frame, text="N/A",style="FrameValueSmall.TLabel")
        self.mean_reversion_label.grid(row=3, column=1, pady=(4, 0), sticky=tk.W)

        # GARCH
        garch_frame = ttk.Frame(f4, style="White.TFrame", padding="8")
        garch_frame.grid(row=0, column=0, sticky="ew", pady=(0, 0))
        ttk.Label(garch_frame, text="GARCH Analysis",style="FrameTitle.TLabel").grid(row=0, column=0, columnspan=3,sticky=tk.W, pady=(0, 4))
        self.predict_button = ttk.Button(garch_frame, text="Predict Volatility",command=self.garch_analysis, state="disabled")
        self.predict_button.grid(row=1, column=0, padx=(0, 10), pady=(0, 6), sticky=tk.W)
        rmse_inner = ttk.Frame(garch_frame)
        rmse_inner.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(rmse_inner, text="RMSE: ",style="FrameLabel.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.rmse_label = ttk.Label(rmse_inner, text="N/A", style="FrameValueBig.TLabel")
        self.rmse_label.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(garch_frame, text="Tomorrow's Regime:",style="FrameLabel.TLabel").grid(row=2, column=0, padx=(0, 4), pady=(4, 0), sticky=tk.W)
        self.tom_regime_label = ttk.Label(garch_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.tom_regime_label.grid(row=2, column=1, pady=(4, 0), sticky=tk.W)
        ttk.Label(garch_frame, text="Tomorrow's 30d RV:", style="FrameLabel.TLabel").grid(row=3, column=0, padx=(0, 4), pady=(4, 0), sticky=tk.W)
        self.tom_rv_label = ttk.Label(garch_frame, text="N/A", style="FrameValueSmall.TLabel")
        self.tom_rv_label.grid(row=3, column=1, pady=(4, 0), sticky=tk.W)
        graph_frame = ttk.Frame(main_frame, style="White.TFrame", padding="10")
        graph_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        self.grap, (self.g1, self.g2, self.g3, self.g4) = plt.subplots(1, 4, figsize=(19, 5.5))
        self.canv = FigureCanvasTkAgg(self.grap, graph_frame)
        self.canv.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        

    def data_query(self):
        symbol = self.tick_var.get().strip().upper()

        if not symbol.endswith('.JK'):
            symbol = f"{symbol}.JK"
        else:
            symbol = symbol
        duration = self.tick_dur.get().lower()

        try:
            extractor = YahooExtract()
            success = extractor.fetch(symbol, duration)

            if success:
                self.stock_data = pd.DataFrame(extractor.output_data.copy())
                self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
                self.query_button.config(state="disabled")
                button_dis = [self.dequery_button, self.analyze_button, self.vol_button, self.predict_button]
                for b in button_dis:
                    b.config(state="normal")
                self.return_data = self.stock_data[['date']].copy()
                self.return_maker()

            else:
                print(f"Failed to fetch data for {symbol}")
        except ValueError as e:
            print(f"Input error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def clear_data(self):
        try:
            self.stock_data = None
            self.return_data = None
            self.volatility_data = None
            self.current_volatility = None
            self.x_vol = None
            self.vol_predict_df = None
            self.q_mod = None
            self.p_mod = None
            self.Rmse = None
            self.query_button.config(state="normal")
            button_dis = [self.dequery_button, self.analyze_button, self.vol_button, self.predict_button]
            for b in button_dis:
                b.config(state="disabled")
            for label in [self.daily_return_label, self.annual_return_label,
                      self.current_daily_return_label, self.current_annual_return_label,
                      self.min_return_label, self.max_return_label,
                      self.current_daily_rv, self.vol_mean_label,
                      self.vol_min_label, self.vol_max_label, 
                      self.regime_label, self.percentile_label,
                      self.mean_reversion_label, self.rmse_label,
                      self.tom_regime_label, self.tom_rv_label]:
                label.config(text="N/A", foreground="black")
            
            for ax in [self.g1, self.g2, self.g3, self.g4]:
                ax.clear()
            self.canv.draw_idle()

        except AttributeError:
            print("No data to clear.")

    def return_maker(self):
        
        if self.stock_data is None:
            print("No Returns Gained")
            return
        self.return_data['daily returns'] = (self.stock_data['close'] / self.stock_data['close'].shift(1))-1
        self.return_data['daily log returns'] = np.log(self.stock_data['close'] / self.stock_data['close'].shift(1))
        self.return_data = self.return_data.dropna()
        self.return_data['annualized log return'] = self.return_data['daily log returns']* self.annualization    


    def returns_analysis(self):
        if self.return_data is not None:
            mean_daily_return = self.return_data['daily returns'].mean()
            current_daily_return = self.return_data['daily returns'].iloc[-1]
            min_val  = self.return_data['daily returns'].min()
            max_val  = self.return_data['daily returns'].max()
            mean_annualized_return = ((1 + mean_daily_return)**self.annualization)-1
            current_annual_return = ((self.stock_data['close'].iloc[-1]/self.stock_data['close'].iloc[0])**(1/(len(self.stock_data)/self.annualization)))-1
            self.daily_return_label.config(text=f"{mean_daily_return*100:.2f}%")
            self.current_daily_return_label.config(text=f"{current_daily_return*100:.2f}%")
            self.min_return_label.config(text=f"{min_val*100:.2f}%")
            self.max_return_label.config(text=f"{max_val*100:.2f}%")
            self.annual_return_label.config(text=f"{mean_annualized_return*100:.2f}%")
            self.current_annual_return_label.config(text=f"{current_annual_return*100:.2f}%")
            self.analyze_button.config(state="disabled")
            self.descriptive_graph()
        else:
            labels = [
            self.daily_return_label, self.current_daily_return_label, self.annual_return_label, 
            self.current_annual_return_label, self.min_return_label, 
            self.max_return_label
             ]
            for label in labels:
                label.config(text="N/A")


    def descriptive_graph(self):
        if self.return_data is None:
            print("Return Does Not Exist, Graph Cannot be Made")
            return
        self.g1.clear()
        colors = ['green' if r > 0 else ('black' if r == 0 else 'red') for r in self.return_data['daily returns']]
        self.g1.vlines(self.return_data['date'],0,self.return_data['daily returns']*100, colors=colors, lw=0.5)
        max_val = max(abs(self.return_data['daily returns'].min()*100), abs(self.return_data['daily returns'].max()*100))
        self.g1.set_ylim(-max_val * 1.1, max_val * 1.1)
        self.g1.grid(True, linestyle='--', alpha=0.3)
        self.g1.axhline(0, color='black', lw=1, alpha=0.5)
        self.g1.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.g1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.grap.autofmt_xdate()
        self.g1.set_title(f"Daily Returns of {self.tick_var.get()}")
        self.g1.set_xlabel("Date")
        self.g1.set_ylabel("Daily Returns (%)")
        self.grap.tight_layout()
        self.canv.draw()

    def vol_analysis(self):
        if self.return_data is None:
            print("Analysis Cannot Be Done, Lack of Returns")
            return
        self.volatility_data = self.return_data[['date']].copy()
        self.volatility_data['hist vol 30d'] = self.return_data['daily log returns'].rolling(window=30, min_periods=1).std() * np.sqrt(self.annualization)  
        self.volatility_data['hv percentile'] = self.volatility_data['hist vol 30d'].rolling(window=self.annualization).rank(pct=True)
        self.volatility_data = self.volatility_data.dropna()
        if self.volatility_data.empty:
            self.volatility_data = None
            return
    
        self.vol_button.config(state="disabled")
        self.Realized_volatility_update()

    def Realized_volatility_update(self):
        if self.volatility_data is not None:
            self.current_volatility = self.volatility_data['hist vol 30d'].iloc[-1]
            min_vol = self.volatility_data['hist vol 30d'].min()
            max_vol = self.volatility_data['hist vol 30d'].max()
            mean_vol = self.volatility_data['hist vol 30d'].mean()
            pct1 = np.percentile(self.volatility_data['hist vol 30d'], 25)
            pct2 = np.percentile(self.volatility_data['hist vol 30d'], 75)
            self.current_daily_rv.config(text=f"{self.current_volatility*100:.2f}%")
            if self.current_volatility > pct2:
                self.current_daily_rv.config(foreground="red")
            elif self.current_volatility < pct1:
                self.current_daily_rv.config(foreground="blue")
            else:
                self.current_daily_rv.config(foreground="black")
            self.vol_mean_label.config(text=f"{mean_vol:.3f}")
            self.vol_min_label.config(text=f"{min_vol:.3f}")
            self.vol_max_label.config(text=f"{max_vol:.3f}")
            self.regime_analysis()
        else:
            self.current_daily_rv.config(text="N/A", foreground="black")    
            self.vol_mean_label.config(text="N/A")
            self.vol_min_label.config(text="N/A")
            self.vol_max_label.config(text="N/A")

    def regime_analysis(self):
        if self.volatility_data is None:
            print("Regime cannot be determined")
            return

        current_percentile = self.volatility_data['hv percentile'].iloc[-1]

        regime, color = next(
            (label, color)
            for threshold, label, color in [
                (0.20, "Low HVOL",           "navy"),
                (0.33, "Below Average HVOL", "blue"),
                (0.66, "Average HVOL",       "black"),
                (0.80, "Above Average HVOL", "red"),
                (1.01, "High HVOL",          "darkred"),
            ]
            if current_percentile < threshold
        )

        reversion, rcolor = next(
            (text, color)
            for threshold, text, color in [
                (0.20, "EXPECT MEAN REVERSION UP",   "red"),
                (0.80, "UNCERTAIN",                  "black"),
                (1.01, "EXPECT MEAN REVERSION DOWN", "green"),
            ]
            if current_percentile < threshold
        )

        self.regime_label.config(text=regime, foreground=color)
        self.percentile_label.config(text=f"{current_percentile:.1%}")
        self.mean_reversion_label.config(text=reversion, foreground=rcolor)
        self.volatility_graph()
    
    def volatility_graph(self):
        if self.volatility_data is None:
            print("Volatility Does Not Exist, Graph Cannot be Made")
            return
        forward_vol = self.volatility_data['hist vol 30d'].shift(-30)
        vol_graph_df = pd.DataFrame({
            'dates' : self.volatility_data['date'].values,
            'current hist vol' : self.volatility_data['hist vol 30d'].values,
            'forward vol' : forward_vol.values,
            'diff_vol' : forward_vol - self.volatility_data['hist vol 30d'].values,
            'percentile vol' : self.volatility_data['hv percentile'].values,
            'current 200d vol': self.return_data['daily log returns'].rolling(window=200, min_periods=1).std().mul(np.sqrt(self.annualization)).reindex(self.volatility_data.index).values
        })
        vol_graph_df = vol_graph_df.dropna()

        if len(vol_graph_df) < 30:
            print("Insufficient Data")
            return
        
        slope_f, intercept_f, r_value_f, p_value_f, std_err_f = stats.linregress(
            vol_graph_df['current hist vol'], 
            vol_graph_df['forward vol']
        )

        if slope_f != 1:
            interception_x = intercept_f / (1-slope_f)
        else:
            interception_x = vol_graph_df['current hist vol'].median()

        high_vol = vol_graph_df['current hist vol']>interception_x
        low_vol = vol_graph_df["current hist vol"]<=interception_x

        if high_vol.sum() > 10:
            slope_high, intercept_high, r_value_high, p_value_high, std_err_high = stats.linregress(vol_graph_df.loc[high_vol, 'current hist vol'], vol_graph_df.loc[high_vol, 'diff_vol'])
        else:
            slope_high = intercept_high = r_value_high = p_value_high = std_err_high = None

        if low_vol.sum() > 10:
            slope_low, intercept_low, r_value_low, p_value_low, std_err_low = stats.linregress(vol_graph_df.loc[low_vol, 'current hist vol'], vol_graph_df.loc[low_vol, 'diff_vol'])
        else:
            slope_low = intercept_low = r_value_low = p_value_low = std_err_low = None

        gs = [self.g2, self.g3]
        for g in gs:
            g.clear()
        
        self.g2.scatter(vol_graph_df.loc[high_vol, 'current hist vol'], vol_graph_df.loc[high_vol, 'diff_vol'], alpha=.6, s=20, color="red", label="High Vol Regime")
        self.g2.scatter(vol_graph_df.loc[low_vol, 'current hist vol'], vol_graph_df.loc[low_vol, 'diff_vol'], alpha=.6, s=20, color="blue", label="Low Vol Regime")

        if slope_high is not None:
            x_high = vol_graph_df.loc[high_vol, 'current hist vol']
            if len(x_high) > 0:
                x_range_high = np.linspace(x_high.min(), x_high.max(), 100)
                y_predhigh = x_range_high*slope_high+intercept_high
                self.g2.plot(x_range_high, y_predhigh, 'r-', linewidth=2, label=f"Regression High Reg R^2={r_value_high**2:.3f}")

        if slope_low is not None:
            x_low = vol_graph_df.loc[low_vol, 'current hist vol']
            if len(x_low) > 0:
                x_range_low = np.linspace(x_low.min(), x_low.max(), 100)
                y_predlow = x_range_low*slope_low+intercept_low
                self.g2.plot(x_range_low, y_predlow, 'r-', linewidth=2, label=f"Regression Low Reg R^2={r_value_low**2:.3f}")

        self.g2.axhline(y=0, color='k',linestyle='--', linewidth=1, label='No Change (y=0)')
        self.g2.axvline(x = interception_x, color='g', linestyle=':', linewidth=1, alpha=.7, label=f"Regime Split (Vol = {interception_x:.3f})")

        self.g2.set_xlabel("Current Realized Volatility")
        self.g2.set_ylabel("Realized Volatility Difference (Forward - Current)")
        self.g2.set_title("Regime Analysis")
        self.g2.legend(fontsize = 7, loc = 'best')
        self.g2.grid(True, alpha=.3)

        self.g3.plot(vol_graph_df['dates'], vol_graph_df['current hist vol'], label='30D Vol', alpha=0.8, color="blue")
        self.g3.plot(vol_graph_df['dates'], vol_graph_df['current 200d vol'], label='200D Vol', linestyle="--", color='brown')
        vol_q1 = vol_graph_df['current hist vol'].quantile(.25)
        vol_q3 = vol_graph_df['current hist vol'].quantile(.75)
        self.g3.axhline(y=vol_q3, color='black', linestyle = '--', alpha = .7, label = "75th Percentile (30d)" )
        self.g3.axhline(y=vol_q1, color='black', linestyle = '--', alpha = .7, label = "25th Percentile (30d)" )
        self.g3.axhline(y=vol_graph_df['current hist vol'].median(), color='black', linestyle='--', alpha=.7, label = "50th Percentile (30d)")

        if self.current_volatility is not None:
            self.g3.scatter(vol_graph_df['dates'].iloc[-1], self.current_volatility, color='red', s=100, zorder=5, label='current IVOL')

        self.g3.set_xlabel('Date')
        self.g3.set_ylabel('Realized Volatility')
        self.g3.grid(True, alpha=.3)
        self.g3.legend(fontsize = 7, loc = 'best')
        self.g3.set_title('Realized Volatility (Long Term and Time Series Analysis)')
        self.g3.tick_params(axis='x', rotation=45)

        self.grap.tight_layout()
        self.canv.draw()

    def garch_analysis(self):
        np.random.seed(67)
        try:
            if self.return_data is not None:
                x_ret = self.return_data['daily log returns'] 
                self.x_vol = x_ret.rolling(window=30, min_periods=1).std() * np.sqrt(self.annualization)
                split = int(len(x_ret) * 0.7)
                x_train = x_ret[:split]
                x_test = self.x_vol[split:]

                def garch_estimator(x):
                    def model_fit(p,q):
                        model = arch_model(x, vol='GARCH', mean='zero', p=p, q=q).fit(disp='off')
                        score = model.bic
                        return score, model
                    p,q=1,1
                    current_score, current_model = model_fit(p,q)
                    while True:
                        neighbors = [(p+1,q),(p-1,q),(p,q+1),(p,q-1)]
                        neighbors = [(pp,qq) for pp,qq in neighbors if pp>=1 and qq >=1]

                        best_score, best_model, best_p, best_q = current_score, current_model, p, q

                        for pp, qq in neighbors:
                            score, model = model_fit(pp,qq)
                            if score < best_score:
                                best_score, best_model = score, model
                                best_p, best_q = pp,qq
                                
                        if best_score < current_score:
                            current_score, current_model = best_score, best_model
                            p, q = best_p, best_q
                        else:
                            break

                    return p,q,current_model
                
                self.p_mod, self.q_mod, model_mod = garch_estimator(x_train)
                forecast_variance = model_mod.forecast(horizon=1).variance.values.flatten()
                x_train_predict = np.sqrt(forecast_variance) * np.sqrt(self.annualization)
                x_test_first = x_test.values[:len(x_train_predict)]
                def rmse(pred, real):
                    return ((sum((pred-real)**2)/len(pred)))**0.5
                self.Rmse = rmse(x_train_predict,x_test_first)
                model_predict = arch_model(x_ret, vol='GARCH', mean='zero', p=self.p_mod, q=self.q_mod).fit(disp='off')
                forecast_obj = model_predict.forecast(horizon=30, method="simulation", simulations=1000)
                sim_variance = forecast_obj.simulations.variances[-1]
                sim_vol = np.sqrt(sim_variance) * np.sqrt(self.annualization)
                self.vol_predict_df = pd.DataFrame({
                    'lower': np.percentile(sim_vol, 2.5, axis=0),
                    'mean':  np.percentile(sim_vol, 50,  axis=0),
                    'upper': np.percentile(sim_vol, 97.5, axis=0)
                }, index=[f'h{i}' for i in range(1, 31)])
                self.predict_button.config(state="disabled")
                self.root.after(0, self.garch_update)
            else:
                print("Return Not Available, GARCH Cannot Be Calculated")
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.log_message(f"GARCH error: {err}"))

    def garch_update(self):
        if self.return_data is None:
            print("Return Not Available, Cannot Be Updated")
            return
        if self.x_vol is not None and self.vol_predict_df is not None:
            combined_vol = pd.concat([self.x_vol, self.vol_predict_df['mean']], axis=0, ignore_index=True)
            future_percentile = combined_vol.rolling(window=self.annualization).rank(pct=True).iloc[-30].item()
            if np.isnan(future_percentile):
                self.log_message("Insufficient data for volatility prediction — try 2y or longer")
                return
            tom_rv = self.vol_predict_df['mean'].iloc[0]
            regime, color = next(
                (label, color)
                for threshold, label, color in [
                    (0.20, "Low HVOL",           "dark blue"),
                    (0.33, "Below Average HVOL", "blue"),
                    (0.66, "Average HVOL",       "black"),
                    (0.80, "Above Average HVOL", "red"),
                    (1.01, "High HVOL",          "dark red"),
                ]
                if future_percentile < threshold
            )

            self.tom_regime_label.config(text=regime, foreground=color)
            self.tom_rv_label.config(text=f"{tom_rv*100:.2f}%")
            if self.Rmse is not None:
                self.rmse_label.config(text=f"{self.Rmse:.6f}")
                self.garch_graph()
            else:
                print("RMSE Cannot be Calculated")
        else:
            print("Cannot get future Volatility")

        
    def garch_graph(self):
        if self.vol_predict_df is not None:
            self.g4.clear()
            x_vol_tail = self.x_vol.iloc[-60:]
            dates_tail = self.stock_data['date'].iloc[-60:]
            n = len(x_vol_tail)
            time_srg_df = pd.DataFrame({
                'dates': pd.concat([dates_tail.reset_index(drop=True),pd.Series(dates_tail.iloc[-1] + pd.to_timedelta(range(1, 31), unit='D'))], ignore_index=True),
                'initial vol':     list(x_vol_tail.values)          + [np.nan] * 30,
                'lower pred vol':  [np.nan] * n + list(self.vol_predict_df['lower'].values),
                'pred vol':        [np.nan] * n + list(self.vol_predict_df['mean'].values),
                'upper pred vol':  [np.nan] * n + list(self.vol_predict_df['upper'].values),
            })
            self.g4.plot(time_srg_df['dates'], time_srg_df['initial vol'], label="Realized Volatility 30d", alpha=0.8, color="blue")
            self.g4.plot(time_srg_df['dates'], time_srg_df['pred vol'], label="Predicted Mean", alpha=0.8, color="orange")
            self.g4.plot(time_srg_df['dates'], time_srg_df['lower pred vol'], linestyle="--", color="red", label="Lower 2.5%")
            self.g4.plot(time_srg_df['dates'], time_srg_df['upper pred vol'], linestyle="--", color="green", label="Upper 97.5%")
            self.g4.set_xlabel('Date')
            self.g4.set_ylabel('Realized Volatility')
            self.g4.grid(True, alpha=.3)
            self.g4.legend()
            self.g4.set_title(f'Realized Volatility Prediction GARCH({self.p_mod},{self.q_mod})')
            self.g4.tick_params(axis='x', rotation=45)
            self.grap.tight_layout()
            self.canv.draw()
        else:
            print("No Predictions To Be Mapped")


def main():
    root = tk.Tk()
    app = RealizedVolDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()