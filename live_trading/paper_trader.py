import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils.logger import get_logger
from data_pipeline.market_data_collector import MarketDataCollector
from risk_management.risk_manager import RiskManager

logger = get_logger(__name__)

class PaperTrader:
    """
    Live Paper Trading Engine.
    Simulates real-time trading against live market data streams.
    Tracks virtual PnL, open positions, and strategy signals.
    """
    def __init__(self, symbol: str, initial_balance: float = 10000.0):
        self.symbol = symbol
        self.balance = initial_balance
        self.position = 0.0  # Amount of asset held
        self.entry_price = 0.0
        self.trade_log: List[Dict[str, Any]] = []
        self.risk_manager = RiskManager(portfolio_value=initial_balance)
        self.collector = MarketDataCollector(symbols=[symbol])

    def run_tick(self, signal: str, current_price: float):
        """Processes a single 'tick' or time-step in paper trading."""
        timestamp = pd.Timestamp.now().isoformat()
        
        if signal == "BUY" and self.position == 0:
            # Check risk before entry
            risk_decision = self.risk_manager.check_risk(current_price)
            if risk_decision.approved:
                qty = (self.balance * risk_decision.position_size) / current_price
                self.balance -= qty * current_price
                self.position = qty
                self.entry_price = current_price
                self._log_trade("BUY", qty, current_price, timestamp)
                logger.info(f"🚀 [PAPER] BUY {qty:.4f} {self.symbol} @ {current_price:.2f}")

        elif signal == "SELL" and self.position > 0:
            pnl = (current_price - self.entry_price) * self.position
            self.balance += self.position * current_price
            self._log_trade("SELL", self.position, current_price, timestamp, pnl)
            logger.info(f"💰 [PAPER] SELL {self.symbol} @ {current_price:.2f} | PnL: ${pnl:.2f}")
            self.position = 0
            self.entry_price = 0

    def _log_trade(self, side: str, qty: float, price: float, timestamp: str, pnl: float = 0.0):
        self.trade_log.append({
            "timestamp": timestamp,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "balance": self.balance
        })

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state of the paper trader."""
        return {
            "symbol": self.symbol,
            "balance": self.balance,
            "position": self.position,
            "n_trades": len(self.trade_log),
            "trade_history": self.trade_log[-5:]
        }
