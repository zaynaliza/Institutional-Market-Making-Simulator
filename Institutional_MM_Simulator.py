
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration matching institutional setups"""
    
    # Simulation
    STEPS = 10000
    START_PRICE = 100.0
    TICK_SIZE = 0.01
    
    # Market Parameters
    ANNUAL_VOL = 0.25
    SPREAD_BPS = 5
    
    # Risk Limits
    MAX_INVENTORY = 100
    MAX_DRAWDOWN_PCT = 0.05
    
    # Strategy Parameters
    INVENTORY_SKEW_STRENGTH = 0.20
    ADVERSE_SELECTION_THRESHOLD = 0.60
    MIN_EDGE_BPS = 0.4
    
    # Fees
    MAKER_REBATE = 0.0002
    TAKER_FEE = 0.0003
    
    # Alpha Model
    USE_ML_ALPHA = True
    ALPHA_DECAY_HALFLIFE = 10
    
    # Market Microstructure
    LOB_DEPTH_LEVELS = 5
    AVG_ORDER_SIZE = 100


# =============================================================================
# LIMIT ORDER BOOK
# =============================================================================

@dataclass
class Order:
    """Represents a single limit order in the book"""
    price: float
    size: int
    is_bid: bool
    timestamp: int
    
class LimitOrderBook:
    """Full limit order book with price-time priority"""
    
    def __init__(self, mid_price: float):
        self.bids = {}
        self.asks = {}
        self.mid_price = mid_price
        self.timestamp = 0
        
    def update_random(self, volatility: float, flow_imbalance: float):
        """Simulate realistic LOB dynamics"""
        self.timestamp += 1
        
        natural_spread = max(Config.TICK_SIZE * 2, volatility * Config.SPREAD_BPS)
        
        self.bids.clear()
        self.asks.clear()
        
        for level in range(Config.LOB_DEPTH_LEVELS):
            bid_price = self.mid_price - natural_spread/2 - level * Config.TICK_SIZE
            bid_size = int(Config.AVG_ORDER_SIZE * (1 + np.random.exponential(0.5)))
            self.bids[round(bid_price, 2)] = bid_size
            
            ask_price = self.mid_price + natural_spread/2 + level * Config.TICK_SIZE
            ask_size = int(Config.AVG_ORDER_SIZE * (1 + np.random.exponential(0.5)))
            if flow_imbalance > 0:
                ask_size = int(ask_size * (1 + flow_imbalance))
            else:
                bid_size = int(bid_size * (1 - flow_imbalance))
            self.asks[round(ask_price, 2)] = ask_size
    
    def get_best_bid_ask(self) -> Tuple[float, float]:
        best_bid = max(self.bids.keys()) if self.bids else self.mid_price - 0.05
        best_ask = min(self.asks.keys()) if self.asks else self.mid_price + 0.05
        return best_bid, best_ask
    
    def get_spread(self) -> float:
        bid, ask = self.get_best_bid_ask()
        return ask - bid
    
    def get_order_imbalance(self) -> float:
        """Order Book Imbalance - key alpha signal"""
        total_bid_size = sum(self.bids.values())
        total_ask_size = sum(self.asks.values())
        
        if total_bid_size + total_ask_size == 0:
            return 0.0
        
        return (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
    
    def get_microprice(self) -> float:
        """Volume-weighted mid price"""
        bid, ask = self.get_best_bid_ask()
        bid_size = self.bids.get(bid, 1)
        ask_size = self.asks.get(ask, 1)
        
        return (bid * ask_size + ask * bid_size) / (bid_size + ask_size)


# =============================================================================
# ADVERSE SELECTION DETECTOR
# =============================================================================

class AdverseSelectionMonitor:
    """VPIN-based toxic flow detection"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.buy_volume = deque(maxlen=lookback)
        self.sell_volume = deque(maxlen=lookback)
        self.price_changes = deque(maxlen=lookback)
        
    def update(self, trade_side: str, volume: int, price_change: float):
        if trade_side == 'buy':
            self.buy_volume.append(volume)
            self.sell_volume.append(0)
        else:
            self.buy_volume.append(0)
            self.sell_volume.append(volume)
        
        self.price_changes.append(abs(price_change))
    
    def get_vpin(self) -> float:
        """Calculate VPIN score [0, 1]"""
        if len(self.buy_volume) < self.lookback // 2:
            return 0.0
        
        total_buy = sum(self.buy_volume)
        total_sell = sum(self.sell_volume)
        total_volume = total_buy + total_sell
        
        if total_volume == 0:
            return 0.0
        
        imbalance = abs(total_buy - total_sell) / total_volume
        return imbalance
    
    def get_realized_volatility(self) -> float:
        if len(self.price_changes) < 10:
            return Config.ANNUAL_VOL / np.sqrt(252 * 28800)
        
        return np.std(list(self.price_changes))


# =============================================================================
# MARKET SIMULATOR - FIXED REGIME DETECTION
# =============================================================================

class MarketSimulator:
    """Market simulator with working regime detection"""
    
    def __init__(self):
        self.lob = LimitOrderBook(Config.START_PRICE)
        self.current_vol = Config.ANNUAL_VOL / np.sqrt(252 * 28800)
        self.regime = 'normal'
        self.step_count = 0
        
        self.last_trade_price = Config.START_PRICE
        self.momentum = 0.0
        
        # FIXED: Calculate dynamic regime thresholds (LOWERED TO ACTUALLY TRIGGER)
        self.base_vol = Config.ANNUAL_VOL / np.sqrt(252 * 28800)
        self.high_vol_threshold = self.base_vol * 1.1  # Only 10% above base
        self.trending_threshold = self.base_vol * 0.8  # Lower threshold for momentum
        
    def step(self) -> dict:
        """Simulate one market tick with proper regime detection"""
        self.step_count += 1
        
        # 1. VOLATILITY CLUSTERING (GARCH) - ENHANCED FOR MORE VARIATION
        vol_shock = np.random.normal(0, self.current_vol)
        
        # More aggressive volatility updates with occasional spikes
        vol_innovation = 0.15 * abs(vol_shock) + 0.01 * np.random.normal(0, self.base_vol * 0.2)
        
        # Add occasional vol spikes to trigger high_vol regime
        if np.random.random() < 0.05:  # 5% chance of vol spike
            vol_innovation += self.base_vol * 0.5
        
        self.current_vol = 0.80 * self.current_vol + vol_innovation
        
        # Allow wider range for volatility
        self.current_vol = np.clip(self.current_vol, self.base_vol * 0.3, self.base_vol * 5.0)
        
        # 2. REGIME DETECTION - WITH WORKING THRESHOLDS
        if self.current_vol > self.high_vol_threshold:
            self.regime = 'high_vol'
        elif abs(self.momentum) > self.trending_threshold:
            self.regime = 'trending'
        else:
            self.regime = 'normal'
        
        # 3. PRICE DYNAMICS - Enhanced for regime variation
        random_component = np.random.normal(0, self.current_vol)
        momentum_component = self.momentum * 0.4
        mean_reversion = -(self.lob.mid_price - Config.START_PRICE) * 0.0001
        
        price_change = random_component + momentum_component + mean_reversion
        self.lob.mid_price += price_change
        
        # Update momentum with more persistence
        self.momentum = 0.75 * self.momentum + 0.25 * price_change
        
        # 4. ORDER FLOW GENERATION
        informed_prob = 0.3 if self.regime == 'trending' else 0.15
        
        buy_flow = False
        sell_flow = False
        flow_type = 'uninformed'
        
        if np.random.random() < 0.4:
            if np.random.random() < informed_prob:
                flow_type = 'informed'
                if price_change > 0:
                    buy_flow = True
                else:
                    sell_flow = True
            else:
                if np.random.random() < 0.5:
                    buy_flow = True
                else:
                    sell_flow = True
        
        # 5. UPDATE ORDER BOOK
        flow_imbalance = 0.3 if buy_flow else (-0.3 if sell_flow else 0.0)
        self.lob.update_random(self.current_vol, flow_imbalance)
        
        # 6. ALPHA SIGNAL
        obi = self.lob.get_order_imbalance()
        alpha = obi * self.current_vol * 100 + self.momentum * 0.5
        
        return {
            'mid_price': self.lob.mid_price,
            'microprice': self.lob.get_microprice(),
            'volatility': self.current_vol,
            'buy_flow': buy_flow,
            'sell_flow': sell_flow,
            'flow_type': flow_type,
            'alpha': alpha,
            'regime': self.regime,
            'spread': self.lob.get_spread(),
            'order_imbalance': obi,
            'best_bid': self.lob.get_best_bid_ask()[0],
            'best_ask': self.lob.get_best_bid_ask()[1]
        }


# =============================================================================
# MARKET MAKING STRATEGY
# =============================================================================

class InstitutionalMarketMaker:
    """Market making strategy with Avellaneda-Stoikov framework"""
    
    def __init__(self):
        self.inventory = 0
        self.cash = 0.0
        
        self.pnl_spread_capture = 0.0
        self.pnl_alpha = 0.0
        self.pnl_adverse_selection = 0.0
        self.pnl_inventory_risk = 0.0
        self.pnl_rebates = 0.0
        
        self.adverse_selection_monitor = AdverseSelectionMonitor()
        self.peak_pnl = 0.0
        self.is_risk_off = False
        
        self.trade_history = []
        self.quote_history = []
        self.metrics = []
        
    def calculate_quotes(self, market_state: dict) -> Tuple[Optional[float], Optional[float]]:
        """Calculate where to place bid and ask"""
        
        # 1. FAIR VALUE
        fair_value = market_state['microprice']
        if Config.USE_ML_ALPHA:
            fair_value += market_state['alpha'] * 0.5
        
        # 2. INVENTORY PENALTY
        inventory_penalty = self.inventory * Config.INVENTORY_SKEW_STRENGTH * market_state['volatility']
        reservation_price = fair_value - inventory_penalty
        
        # 3. DYNAMIC SPREAD
        base_spread = market_state['volatility'] * 2.0
        
        vpin = self.adverse_selection_monitor.get_vpin()
        toxicity_multiplier = 1.0
        if vpin > Config.ADVERSE_SELECTION_THRESHOLD:
            toxicity_multiplier = 1.5
            if vpin > 0.8:
                return None, None
        
        inventory_multiplier = 1.0 + (abs(self.inventory) / Config.MAX_INVENTORY) * 0.5
        
        spread = base_spread * toxicity_multiplier * inventory_multiplier
        spread = max(spread, market_state['spread'] + Config.TICK_SIZE)
        
        # 4. CALCULATE BID/ASK
        bid = np.round((reservation_price - spread/2) / Config.TICK_SIZE) * Config.TICK_SIZE
        ask = np.round((reservation_price + spread/2) / Config.TICK_SIZE) * Config.TICK_SIZE
        
        # 5. EDGE CHECK
        bid_edge = market_state['microprice'] - bid
        ask_edge = ask - market_state['microprice']
        
        min_edge = Config.MIN_EDGE_BPS * market_state['microprice'] / 10000
        
        final_bid = bid if bid_edge > min_edge else None
        final_ask = ask if ask_edge > min_edge else None
        
        # 6. POSITION LIMITS
        if self.inventory >= Config.MAX_INVENTORY:
            final_bid = None
        if self.inventory <= -Config.MAX_INVENTORY:
            final_ask = None
        
        # 7. RISK-OFF MODE
        if self.is_risk_off:
            return None, None
        
        return final_bid, final_ask
    
    def execute_trades(self, market_state: dict, bid: Optional[float], ask: Optional[float]):
        """Execute trades with more realistic fill probability"""
        
        # MARKET BUY ORDER (hits our ask)
        if market_state['buy_flow'] and ask is not None:
            # FIXED: Lower fill probability (was 0.7, now 0.55)
            if np.random.random() < 0.55:
                self.inventory -= 1
                self.cash += ask
                
                self.cash += Config.MAKER_REBATE
                self.pnl_rebates += Config.MAKER_REBATE
                
                immediate_pnl = ask - market_state['microprice']
                self.pnl_spread_capture += immediate_pnl
                
                self.trade_history.append({
                    'step': len(self.metrics),
                    'side': 'sell',
                    'price': ask,
                    'fair_value': market_state['microprice'],
                    'flow_type': market_state['flow_type'],
                    'edge': immediate_pnl
                })
                
                self.adverse_selection_monitor.update('sell', 1, 
                                                      ask - market_state['microprice'])
        
        # MARKET SELL ORDER (hits our bid)
        if market_state['sell_flow'] and bid is not None:
            if np.random.random() < 0.55:
                self.inventory += 1
                self.cash -= bid
                
                self.cash += Config.MAKER_REBATE
                self.pnl_rebates += Config.MAKER_REBATE
                
                immediate_pnl = market_state['microprice'] - bid
                self.pnl_spread_capture += immediate_pnl
                
                self.trade_history.append({
                    'step': len(self.metrics),
                    'side': 'buy',
                    'price': bid,
                    'fair_value': market_state['microprice'],
                    'flow_type': market_state['flow_type'],
                    'edge': immediate_pnl
                })
                
                self.adverse_selection_monitor.update('buy', 1,
                                                      market_state['microprice'] - bid)
    
    def update_pnl(self, market_state: dict, prev_mid: float):
        """Mark-to-market PnL with attribution"""
        price_change = market_state['mid_price'] - prev_mid
        inventory_pnl = self.inventory * price_change
        self.pnl_inventory_risk += inventory_pnl
        
        total_pnl = self.cash + (self.inventory * market_state['mid_price'])
        
        if total_pnl > self.peak_pnl:
            self.peak_pnl = total_pnl
        
        drawdown = (self.peak_pnl - total_pnl) / max(abs(self.peak_pnl), 1)
        if drawdown > Config.MAX_DRAWDOWN_PCT:
            self.is_risk_off = True
        
        return total_pnl
    
    def step(self, market_state: dict, prev_mid: float):
        """One iteration of the strategy"""
        
        bid, ask = self.calculate_quotes(market_state)
        self.execute_trades(market_state, bid, ask)
        total_pnl = self.update_pnl(market_state, prev_mid)
        
        self.metrics.append({
            'mid_price': market_state['mid_price'],
            'microprice': market_state['microprice'],
            'inventory': self.inventory,
            'total_pnl': total_pnl,
            'spread_pnl': self.pnl_spread_capture,
            'inventory_pnl': self.pnl_inventory_risk,
            'rebate_pnl': self.pnl_rebates,
            'volatility': market_state['volatility'],
            'vpin': self.adverse_selection_monitor.get_vpin(),
            'regime': market_state['regime'],
            'bid': bid if bid else np.nan,
            'ask': ask if ask else np.nan,
            'spread': market_state['spread'],
            'obi': market_state['order_imbalance']
        })
        
        self.quote_history.append({
            'step': len(self.metrics) - 1,
            'bid': bid,
            'ask': ask,
            'mid': market_state['mid_price']
        })


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest() -> Tuple[InstitutionalMarketMaker, pd.DataFrame]:
    """Execute full simulation"""
    print("=" * 70)
    print("INSTITUTIONAL MARKET MAKING BACKTEST")
    print("=" * 70)
    print(f"Steps: {Config.STEPS:,}")
    print(f"Annual Volatility: {Config.ANNUAL_VOL:.1%}")
    print(f"Max Inventory: {Config.MAX_INVENTORY}")
    print(f"Adverse Selection Threshold: {Config.ADVERSE_SELECTION_THRESHOLD}")
    print("=" * 70)
    
    market = MarketSimulator()
    strategy = InstitutionalMarketMaker()
    
    prev_mid = Config.START_PRICE
    
    for step in range(Config.STEPS):
        if step % 1000 == 0:
            print(f"Progress: {step/Config.STEPS:.1%}", end='\r')
        
        market_state = market.step()
        strategy.step(market_state, prev_mid)
        prev_mid = market_state['mid_price']
    
    print(f"Progress: 100.0%")
    print("\nBacktest complete!")
    
    df = pd.DataFrame(strategy.metrics)
    return strategy, df


# =============================================================================
# ANALYTICS - FIXED PLOTS
# =============================================================================

def plot_institutional_dashboard(strategy: InstitutionalMarketMaker, df: pd.DataFrame):
    """Create professional analytics dashboard with all fixes"""
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    colors = {
        'pnl': '#00FF41',
        'inventory': '#00BFFF',
        'spread': '#FFD700',
        'vpin': '#FF4500',
        'vol': '#9370DB'
    }
    
    # 1. PNL ATTRIBUTION
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['total_pnl'], label='Total PnL', color=colors['pnl'], linewidth=2)
    ax1.fill_between(df.index, df['spread_pnl'], alpha=0.3, color='#00FF41', label='Spread Capture')
    ax1.fill_between(df.index, df['inventory_pnl'], alpha=0.3, color='#FF4500', label='Inventory Risk')
    ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax1.set_title('PNL ATTRIBUTION: Where Did Profits Come From?', 
                  fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(alpha=0.2)
    
    # 2. INVENTORY MANAGEMENT
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['inventory'], color=colors['inventory'], linewidth=1.5)
    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax2.axhline(Config.MAX_INVENTORY, color='red', linestyle='--', alpha=0.5, label='Limit')
    ax2.axhline(-Config.MAX_INVENTORY, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, df['inventory'], alpha=0.3, color=colors['inventory'])
    ax2.set_title('Inventory Management (Mean Reverting)', fontweight='bold', loc='left')
    ax2.set_ylabel('Position (shares)')
    ax2.legend()
    ax2.grid(alpha=0.2)
    
    # 3. ADVERSE SELECTION (VPIN)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['vpin'], color=colors['vpin'], linewidth=1.5)
    ax3.axhline(Config.ADVERSE_SELECTION_THRESHOLD, color='yellow', 
                linestyle='--', alpha=0.7, label='Warning Threshold')
    ax3.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Critical')
    ax3.fill_between(df.index, df['vpin'], alpha=0.3, color=colors['vpin'])
    ax3.set_title('Toxic Flow Detection (VPIN)', fontweight='bold', loc='left')
    ax3.set_ylabel('VPIN Score')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(alpha=0.2)
    
    # 4. VOLATILITY CLUSTERING - FIXED SCALING
    ax4 = fig.add_subplot(gs[1, 2])
    vol_annualized = df['volatility'] * np.sqrt(252 * 28800) * 100
    ax4.plot(vol_annualized, color=colors['vol'], linewidth=1.5)
    ax4.fill_between(df.index, vol_annualized, alpha=0.3, color=colors['vol'])
    ax4.set_title('Realized Volatility (GARCH)', fontweight='bold', loc='left')
    ax4.set_ylabel('Annualized Vol (%)')
    ax4.grid(alpha=0.2)
    
    # 5. DRAWDOWN ANALYSIS
    ax5 = fig.add_subplot(gs[2, 0])
    running_max = df['total_pnl'].cummax()
    drawdown = df['total_pnl'] - running_max
    ax5.fill_between(df.index, drawdown, color='red', alpha=0.4)
    ax5.set_title('Strategy Drawdown Profile', fontweight='bold', loc='left')
    ax5.set_ylabel('Drawdown ($)')
    ax5.grid(alpha=0.2)
    
    # 6. SPREAD DYNAMICS - FIXED
    ax6 = fig.add_subplot(gs[2, 1])
    # Only plot where we have valid quotes
    valid_quotes = ~(df['bid'].isna() | df['ask'].isna())
    if valid_quotes.sum() > 0:
        quoted_spread = ((df['ask'] - df['bid']) * 10000)[valid_quotes]
        ax6.scatter(df.index[valid_quotes], quoted_spread, 
                   color=colors['spread'], s=2, alpha=0.4, label='Quoted Spread')
    ax6.plot(df['spread'] * 10000, color='white', linewidth=1, alpha=0.5, 
             linestyle='--', label='Market Spread')
    ax6.set_title('Spread Management (bps)', fontweight='bold', loc='left')
    ax6.set_ylabel('Spread (bps)')
    ax6.legend()
    ax6.grid(alpha=0.2)
    
    # 7. RETURNS DISTRIBUTION
    ax7 = fig.add_subplot(gs[2, 2])
    returns = df['total_pnl'].diff().dropna()
    ax7.hist(returns, bins=50, color=colors['pnl'], alpha=0.7, edgecolor='white')
    ax7.axvline(returns.mean(), color='yellow', linestyle='--', 
                linewidth=2, label=f'Mean: {returns.mean():.4f}')
    ax7.set_title('Return Distribution', fontweight='bold', loc='left')
    ax7.set_xlabel('Period Return ($)')
    ax7.legend()
    ax7.grid(alpha=0.2)
    
    # 8. REGIME PERFORMANCE - FIXED
    ax8 = fig.add_subplot(gs[3, 0])
    regime_pnl = {}
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        if len(regime_df) > 0:
            regime_pnl[regime] = regime_df['total_pnl'].iloc[-1] - regime_df['total_pnl'].iloc[0]
    
    if regime_pnl:
        pd.Series(regime_pnl).plot(kind='bar', ax=ax8, 
                                   color=[colors['pnl'], colors['vpin'], colors['vol']])
    ax8.set_title('PnL by Market Regime', fontweight='bold', loc='left')
    ax8.set_ylabel('PnL ($)')
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
    ax8.grid(alpha=0.2)
    
    # 9. ORDER BOOK IMBALANCE vs PRICE - FIXED
    ax9 = fig.add_subplot(gs[3, 1])
    ax9_twin = ax9.twinx()
    ax9.plot(df['obi'], color=colors['inventory'], alpha=0.6, label='OBI')
    ax9_twin.plot(df['mid_price'], color='white', alpha=0.4, label='Price')
    ax9_twin.ticklabel_format(style='plain', axis='y')  # FIX: Remove scientific notation
    ax9.set_title('Order Imbalance vs Price', fontweight='bold', loc='left')
    ax9.set_ylabel('OBI', color=colors['inventory'])
    ax9_twin.set_ylabel('Price ($)', color='white')
    ax9.grid(alpha=0.2)
    
    # 10. TRADE ANALYSIS
    ax10 = fig.add_subplot(gs[3, 2])
    trades_df = pd.DataFrame(strategy.trade_history)
    if len(trades_df) > 0:
        informed_trades = trades_df[trades_df['flow_type'] == 'informed']
        uninformed_trades = trades_df[trades_df['flow_type'] == 'uninformed']
        
        ax10.scatter(informed_trades['step'], informed_trades['edge'], 
                    color='red', alpha=0.6, s=20, label='Informed Flow')
        ax10.scatter(uninformed_trades['step'], uninformed_trades['edge'],
                    color='green', alpha=0.6, s=20, label='Uninformed Flow')
        ax10.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax10.set_title('Trade Quality: Edge per Trade', fontweight='bold', loc='left')
        ax10.set_ylabel('Edge ($)')
        ax10.legend()
    ax10.grid(alpha=0.2)
    
    plt.suptitle('INSTITUTIONAL MARKET MAKING DASHBOARD', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def print_performance_summary(strategy: InstitutionalMarketMaker, df: pd.DataFrame):
    """Print performance metrics"""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Returns
    total_pnl = df['total_pnl'].iloc[-1]
    returns = df['total_pnl'].diff().dropna()
    
    # FIXED: More conservative Sharpe calculation
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(min(252, len(returns) / 100))
    else:
        sharpe = 0
    
    # Max Drawdown
    running_max = df['total_pnl'].cummax()
    drawdown = ((df['total_pnl'] - running_max) / running_max.abs()).min()
    
    # Trade statistics
    trades_df = pd.DataFrame(strategy.trade_history)
    num_trades = len(trades_df)
    
    if num_trades > 0:
        avg_edge = trades_df['edge'].mean()
        win_rate = (trades_df['edge'] > 0).sum() / num_trades
        
        informed_trades = trades_df[trades_df['flow_type'] == 'informed']
        informed_edge = informed_trades['edge'].mean() if len(informed_trades) > 0 else 0
    else:
        avg_edge = 0
        win_rate = 0
        informed_edge = 0
    
    # PnL Attribution
    spread_pct = (df['spread_pnl'].iloc[-1] / total_pnl * 100) if total_pnl != 0 else 0
    inventory_pct = (df['inventory_pnl'].iloc[-1] / total_pnl * 100) if total_pnl != 0 else 0
    
    print(f"\n📊 RETURNS")
    print(f"  Total PnL:              ${total_pnl:,.2f}")
    print(f"  Sharpe Ratio:           {sharpe:.2f}")
    print(f"  Max Drawdown:           {drawdown:.2%}")
    
    print(f"\n💰 PNL ATTRIBUTION")
    print(f"  Spread Capture:         ${df['spread_pnl'].iloc[-1]:,.2f} ({spread_pct:.1f}%)")
    print(f"  Inventory P&L:          ${df['inventory_pnl'].iloc[-1]:,.2f} ({inventory_pct:.1f}%)")
    print(f"  Rebates:                ${df['rebate_pnl'].iloc[-1]:,.2f}")
    
    print(f"\n📈 TRADING ACTIVITY")
    print(f"  Total Trades:           {num_trades:,}")
    print(f"  Avg Edge per Trade:     ${avg_edge:.4f}")
    print(f"  Win Rate:               {win_rate:.1%}")
    print(f"  Edge on Informed Flow:  ${informed_edge:.4f}")
    
    print(f"\n⚠️  RISK METRICS")
    print(f"  Max Inventory:          {df['inventory'].abs().max():.0f} shares")
    print(f"  Avg VPIN:               {df['vpin'].mean():.3f}")
    print(f"  Risk-Off Triggered:     {'Yes' if strategy.is_risk_off else 'No'}")
    
    print(f"\n🎯 MARKET CONDITIONS")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime.capitalize()} Regime:      {count/len(df):.1%} of time")
    
    print("\n" + "=" * 70)
    
    # ASSESSMENT
    print("\n🔍 STRATEGY ASSESSMENT:")
    if sharpe > 2:
        print("  ✓ Excellent risk-adjusted returns")
    elif sharpe > 1:
        print("  ✓ Good risk-adjusted returns")
    else:
        print("  ⚠ Risk-adjusted returns need improvement")
    
    if abs(drawdown) < 0.1:
        print("  ✓ Drawdown well controlled")
    else:
        print("  ⚠ High drawdown - review risk management")
    
    if win_rate > 0.55:
        print("  ✓ High win rate - good trade selection")
    else:
        print("  ⚠ Win rate below 55% - check adverse selection")
    
    # FIXED: Correct logic for informed flow edge
    if num_trades > 0 and len(informed_trades) > 0:
        if informed_edge < 0:
            print("  ⚠ Losing money on informed flow - needs improvement")
        else:
            print("  ✓ Positive edge even on informed flow")
    else:
        print("  ✓ Successfully avoiding adverse selection")
    
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Run backtest
    strategy, results_df = run_backtest()
    
    # Print performance metrics
    print_performance_summary(strategy, results_df)
    
    # Generate dashboard
    fig = plot_institutional_dashboard(strategy, results_df)
    plt.show()
    
    # Create output directory
    os.makedirs('sample_results', exist_ok=True)
    
    # Save dashboard as high-quality image
    fig.savefig('sample_results/dashboard_output.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    print(f"📊 Dashboard saved to: sample_results/dashboard_output.png")
    
    # Save results
    results_df.to_csv('sample_results/mm_backtest_results.csv', index=False)
    print(f"💾 Results saved to: mm_backtest_results.csv")
    
    print("\n✅ Analysis complete!")