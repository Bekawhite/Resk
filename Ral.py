import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import hashlib
import uuid
import yfinance as yf
import requests
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import base64
from fpdf import FPDF
import sqlite3
from streamlit_authenticator import Authenticate
import bcrypt
import forex_python.converter as fx
from alpha_vantage.foreignexchange import ForeignExchange
import smtplib
from email.mime.text import MimeText
from twilio.rest import Client
import re
from textblob import TextBlob
import nltk
from transformers import pipeline
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="ALPHA Risk Manager Pro+ | Enterprise Edition",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        border-left: 5px solid #cc0000;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        border-left: 5px solid #cc8400;
    }
    .risk-low {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        border-left: 5px solid #27ae60;
    }
    .alert-box {
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        background-color: #fff3f3;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .prediction-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .copyright {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .kes-currency {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #27ae60;
    }
    .news-positive {
        border-left: 4px solid #28a745;
        background-color: #f8fff9;
        padding: 10px;
        margin: 5px 0;
    }
    .news-negative {
        border-left: 4px solid #dc3545;
        background-color: #fff8f8;
        padding: 10px;
        margin: 5px 0;
    }
    .news-neutral {
        border-left: 4px solid #6c757d;
        background-color: #f8f9fa;
        padding: 10px;
        margin: 5px 0;
    }
    .compliance-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .compliance-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .ai-explanation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class AIRiskExplainer:
    """AI-Powered Risk Explanation System"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.risk_keywords = {
            'high': ['volatility', 'crash', 'drop', 'plunge', 'selloff', 'risk', 'warning', 'bear'],
            'medium': ['fluctuate', 'uncertain', 'mixed', 'caution', 'adjust', 'moderate'],
            'low': ['stable', 'growth', 'rally', 'bull', 'gain', 'positive', 'strong']
        }
    
    def generate_risk_explanation(self, symbol, risk_metrics, news_sentiment=None):
        """Generate natural language explanation of risk factors"""
        
        volatility = risk_metrics.get('volatility', 0)
        var = risk_metrics.get('var_95', 0)
        drawdown = risk_metrics.get('max_drawdown', 0)
        
        # Determine risk level
        if volatility > 0.3 or var > 0.05:
            risk_level = "HIGH"
            urgency = "immediate attention"
        elif volatility > 0.15 or var > 0.02:
            risk_level = "MODERATE" 
            urgency = "close monitoring"
        else:
            risk_level = "LOW"
            urgency = "regular monitoring"
        
        # Generate explanation
        explanation_parts = []
        
        # Volatility explanation
        if volatility > 0.25:
            explanation_parts.append(f"**{symbol} shows elevated volatility** ({volatility:.1%}), indicating significant price fluctuations that require {urgency}.")
        elif volatility > 0.15:
            explanation_parts.append(f"**{symbol} demonstrates moderate volatility** ({volatility:.1%}), suggesting typical market movements that warrant {urgency}.")
        else:
            explanation_parts.append(f"**{symbol} exhibits low volatility** ({volatility:.1%}), reflecting stable price behavior suitable for {urgency}.")
        
        # VaR explanation
        if var > 0.04:
            explanation_parts.append(f"**Value at Risk is concerning** - there's a 5% chance of losing more than KES {var:,.0f} in one day, highlighting significant downside risk.")
        elif var > 0.02:
            explanation_parts.append(f"**Value at Risk is moderate** - potential one-day loss of up to KES {var:,.0f} falls within acceptable ranges for most institutional portfolios.")
        else:
            explanation_parts.append(f"**Value at Risk is minimal** - maximum expected loss of KES {var:,.0f} indicates strong capital preservation characteristics.")
        
        # Drawdown explanation
        if drawdown > 0.15:
            explanation_parts.append(f"**Historical drawdowns have been severe** ({drawdown:.1%} peak-to-trough decline), suggesting vulnerability during market stress periods.")
        elif drawdown > 0.08:
            explanation_parts.append(f"**Moderate historical drawdowns** ({drawdown:.1%}) indicate reasonable resilience with some exposure to market corrections.")
        else:
            explanation_parts.append(f"**Limited historical drawdowns** ({drawdown:.1%}) demonstrate strong defensive characteristics during market downturns.")
        
        # Add news sentiment if available
        if news_sentiment:
            if news_sentiment > 0.1:
                explanation_parts.append("**Recent news sentiment is positive**, potentially offsetting some technical risk factors.")
            elif news_sentiment < -0.1:
                explanation_parts.append("**Negative news sentiment amplifies risk concerns**, warranting enhanced due diligence.")
        
        explanation_parts.append(f"\n**Overall Risk Assessment: {risk_level}** - {self.get_recommendation(risk_level)}")
        
        return "\n\n".join(explanation_parts)
    
    def get_recommendation(self, risk_level):
        """Get AI-generated recommendation based on risk level"""
        recommendations = {
            "HIGH": "Consider reducing position size, implementing hedging strategies, or setting tighter stop-loss limits. Immediate portfolio review recommended.",
            "MODERATE": "Maintain current position with increased monitoring. Consider dollar-cost averaging for additional purchases.",
            "LOW": "Position appears stable. Continue normal monitoring procedures. Suitable for conservative investment mandates."
        }
        return recommendations.get(risk_level, "No specific recommendation available.")
    
    def analyze_news_sentiment(self, news_text):
        """Analyze sentiment of financial news"""
        try:
            if not news_text or len(news_text) < 10:
                return 0.0
            
            # Simple sentiment analysis using TextBlob
            analysis = TextBlob(news_text)
            sentiment = analysis.sentiment.polarity
            
            return sentiment
        except:
            return 0.0

class RealTimeNewsIntegration:
    """Real-time Financial News Integration"""
    
    def __init__(self):
        self.news_api_key = "demo_key"  # In production, use actual API key
        
    def get_financial_news(self, symbols=None, limit=10):
        """Get real-time financial news for symbols"""
        try:
            # Simulated news data - in production, integrate with NewsAPI, Bloomberg, etc.
            sample_news = [
                {
                    'title': 'Central Bank of Kenya Holds Interest Rates Steady',
                    'description': 'The Monetary Policy Committee maintains base rate at 13.0%, citing inflation concerns.',
                    'symbol': 'CBK',
                    'sentiment': -0.2,
                    'source': 'Business Daily Africa',
                    'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'impact': 'HIGH'
                },
                {
                    'title': 'Safaricom Reports Strong Quarterly Earnings',
                    'description': 'Mobile operator exceeds revenue expectations with 15% growth in M-PESA transactions.',
                    'symbol': 'SCOM',
                    'sentiment': 0.7,
                    'source': 'Reuters',
                    'published_at': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                    'impact': 'MEDIUM'
                },
                {
                    'title': 'East African Breweries Expands Regional Operations',
                    'description': 'Company announces new manufacturing facility in Tanzania to capture growing market.',
                    'symbol': 'EABL',
                    'sentiment': 0.5,
                    'source': 'The EastAfrican',
                    'published_at': (datetime.now() - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'),
                    'impact': 'MEDIUM'
                },
                {
                    'title': 'Kenya Airways Secures Government Bailout Package',
                    'description': 'National carrier receives KES 15 billion injection to support operational recovery.',
                    'symbol': 'KQ',
                    'sentiment': 0.3,
                    'source': 'Bloomberg',
                    'published_at': (datetime.now() - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
                    'impact': 'HIGH'
                }
            ]
            
            if symbols:
                filtered_news = [news for news in sample_news if news.get('symbol') in symbols]
                return filtered_news[:limit]
            
            return sample_news[:limit]
            
        except Exception as e:
            st.warning(f"News feed temporarily unavailable: {str(e)}")
            return []

class MobileAlertSystem:
    """Enterprise Mobile Alert System"""
    
    def __init__(self):
        self.twilio_account_sid = "demo_sid"
        self.twilio_auth_token = "demo_token"
        self.twilio_phone = "+1234567890"
    
    def send_sms_alert(self, phone_number, message):
        """Send SMS alert (simulated)"""
        try:
            # In production, integrate with Twilio API
            st.success(f"📱 SMS Alert Sent to {phone_number}: {message}")
            return True
        except Exception as e:
            st.error(f"Failed to send SMS: {str(e)}")
            return False
    
    def send_email_alert(self, email, subject, message):
        """Send email alert (simulated)"""
        try:
            # In production, integrate with SMTP server
            st.success(f"📧 Email Alert Sent to {email}: {subject}")
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def check_risk_thresholds(self, portfolio_metrics, user_preferences):
        """Check if any risk thresholds are breached"""
        alerts = []
        
        # VaR threshold check
        if portfolio_metrics.get('var_95', 0) > user_preferences.get('var_threshold', 0.03):
            alerts.append({
                'type': 'VAR_BREACH',
                'message': f"Value at Risk ({portfolio_metrics['var_95']:.2%}) exceeds threshold ({user_preferences['var_threshold']:.2%})",
                'severity': 'HIGH'
            })
        
        # Volatility threshold check
        if portfolio_metrics.get('volatility', 0) > user_preferences.get('volatility_threshold', 0.2):
            alerts.append({
                'type': 'VOLATILITY_BREACH',
                'message': f"Portfolio volatility ({portfolio_metrics['volatility']:.2%}) exceeds threshold ({user_preferences['volatility_threshold']:.2%})",
                'severity': 'MEDIUM'
            })
        
        # Concentration check
        if portfolio_metrics.get('concentration', 0) > user_preferences.get('concentration_threshold', 0.3):
            alerts.append({
                'type': 'CONCENTRATION_BREACH',
                'message': f"Portfolio concentration ({portfolio_metrics['concentration']:.2%}) exceeds threshold ({user_preferences['concentration_threshold']:.2%})",
                'severity': 'MEDIUM'
            })
        
        return alerts

class RegulatoryComplianceDashboard:
    """Central Bank of Kenya Regulatory Compliance Monitoring"""
    
    def __init__(self):
        self.cbk_regulations = {
            'capital_adequacy': 0.08,  # 8% minimum capital adequacy
            'liquidity_coverage': 1.0,  # 100% LCR
            'large_exposure_limit': 0.25,  # 25% single exposure limit
            'stress_test_frequency': 'quarterly',
            'reporting_deadline': 5  # business days after month end
        }
    
    def check_compliance_status(self, portfolio_metrics, institution_data):
        """Check compliance with CBK regulations"""
        compliance_status = {}
        
        # Capital Adequacy Ratio (CAR)
        car = institution_data.get('capital_adequacy_ratio', 0)
        compliance_status['capital_adequacy'] = {
            'required': self.cbk_regulations['capital_adequacy'],
            'actual': car,
            'compliant': car >= self.cbk_regulations['capital_adequacy'],
            'description': 'Minimum Capital Adequacy Ratio (CAR) of 8%'
        }
        
        # Liquidity Coverage Ratio (LCR)
        lcr = institution_data.get('liquidity_coverage_ratio', 0)
        compliance_status['liquidity_coverage'] = {
            'required': self.cbk_regulations['liquidity_coverage'],
            'actual': lcr,
            'compliant': lcr >= self.cbk_regulations['liquidity_coverage'],
            'description': 'Minimum Liquidity Coverage Ratio (LCR) of 100%'
        }
        
        # Large Exposure Limit
        max_exposure = portfolio_metrics.get('max_single_exposure', 0)
        compliance_status['large_exposure'] = {
            'required': self.cbk_regulations['large_exposure_limit'],
            'actual': max_exposure,
            'compliant': max_exposure <= self.cbk_regulations['large_exposure_limit'],
            'description': 'Maximum single exposure limit of 25%'
        }
        
        # Stress Testing
        last_stress_test = institution_data.get('last_stress_test')
        if last_stress_test:
            days_since_test = (datetime.now() - last_stress_test).days
            compliant = days_since_test <= 90  # Quarterly requirement
        else:
            compliant = False
            
        compliance_status['stress_testing'] = {
            'required': 'Quarterly',
            'actual': last_stress_test.strftime('%Y-%m-%d') if last_stress_test else 'Never',
            'compliant': compliant,
            'description': 'Quarterly stress testing requirement'
        }
        
        return compliance_status
    
    def generate_compliance_report(self, compliance_status):
        """Generate regulatory compliance report"""
        report = {
            'overall_compliant': all(status['compliant'] for status in compliance_status.values()),
            'details': compliance_status,
            'next_reporting_date': (datetime.now() + timedelta(days=30)).replace(day=1).strftime('%Y-%m-%d'),
            'regulatory_contact': 'Central Bank of Kenya - Banking Supervision Department'
        }
        return report

class PeerBenchmarking:
    """Peer Institution Benchmarking System"""
    
    def __init__(self):
        self.peer_groups = {
            'tier_1_banks': ['KCB', 'EQTY', 'COOP', 'NCBA', 'SCBK'],
            'insurance': ['JUB', 'BRIT', 'CFCI', 'CIC'],
            'saccos': ['UNGA', 'STIMA', 'AFRICA'],
            'microfinance': ['CARB', 'FAIR']
        }
    
    def get_peer_comparison(self, portfolio_metrics, institution_type):
        """Compare performance against peer institutions"""
        # Simulated peer data - in production, integrate with actual market data
        peer_benchmarks = {
            'tier_1_banks': {
                'avg_return': 0.12,
                'avg_volatility': 0.18,
                'avg_var': 0.025,
                'avg_sharpe': 0.65,
                'top_performer': 'EQTY'
            },
            'insurance': {
                'avg_return': 0.15,
                'avg_volatility': 0.22,
                'avg_var': 0.032,
                'avg_sharpe': 0.58,
                'top_performer': 'JUB'
            },
            'saccos': {
                'avg_return': 0.08,
                'avg_volatility': 0.12,
                'avg_var': 0.018,
                'avg_sharpe': 0.45,
                'top_performer': 'STIMA'
            },
            'microfinance': {
                'avg_return': 0.20,
                'avg_volatility': 0.28,
                'avg_var': 0.045,
                'avg_sharpe': 0.52,
                'top_performer': 'CARB'
            }
        }
        
        benchmark = peer_benchmarks.get(institution_type, peer_benchmarks['tier_1_banks'])
        
        comparison = {}
        for metric, value in portfolio_metrics.items():
            if metric in benchmark:
                peer_avg = benchmark[metric]
                comparison[metric] = {
                    'your_value': value,
                    'peer_average': peer_avg,
                    'percentile': min(100, max(0, (value / peer_avg) * 50 + 50)) if peer_avg != 0 else 50,
                    'outperformance': value - peer_avg
                }
        
        comparison['top_performer'] = benchmark.get('top_performer', 'Unknown')
        comparison['peer_group_size'] = len(self.peer_groups.get(institution_type, []))
        
        return comparison

class CurrencyConverter:
    """Enhanced currency conversion with KES support"""
    
    def __init__(self):
        self.usd_to_kes = 150.0  # Default rate, will be updated from API
        self.update_exchange_rates()
    
    def update_exchange_rates(self):
        """Update exchange rates from API"""
        try:
            # Using forex-python for real exchange rates
            converter = fx.CurrencyRates()
            self.usd_to_kes = converter.get_rate('USD', 'KES')
        except:
            # Fallback to reasonable estimate
            self.usd_to_kes = 150.0
    
    def usd_to_kes_format(self, amount):
        """Convert USD to KES with proper formatting"""
        kes_amount = amount * self.usd_to_kes
        return f"KES {kes_amount:,.2f}"
    
    def format_currency(self, amount, currency='KES'):
        """Format currency with proper symbols"""
        if currency == 'KES':
            return f"KES {amount:,.2f}"
        else:
            return f"${amount:,.2f}"

class EnhancedMLRiskPredictor:
    """Enhanced Machine Learning Risk Forecasting Engine"""
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_features(self, price_df, volume_df, window=30):
        """Prepare enhanced features for ML model"""
        returns = price_df.pct_change().dropna()
        
        features = {}
        for symbol in price_df.columns:
            try:
                # Technical indicators
                returns_series = returns[symbol]
                
                # Enhanced rolling features
                volatility = returns_series.rolling(window=window).std() * np.sqrt(252)
                momentum = price_df[symbol] / price_df[symbol].shift(window) - 1
                rsi = self.calculate_rsi(price_df[symbol], window=14)
                volume_ma = volume_df[symbol].rolling(window=window).mean()
                
                # Additional features for better prediction
                price_ma_ratio = price_df[symbol] / price_df[symbol].rolling(window=50).mean()
                volatility_ratio = volatility / volatility.rolling(window=50).mean()
                
                # Combine features
                feature_df = pd.DataFrame({
                    'volatility': volatility,
                    'momentum': momentum,
                    'rsi': rsi,
                    'volume_ma': volume_ma,
                    'price_ma_ratio': price_ma_ratio,
                    'volatility_ratio': volatility_ratio,
                    'returns_lag1': returns_series.shift(1),
                    'returns_lag5': returns_series.shift(5),
                    'returns_lag10': returns_series.shift(10)
                }).fillna(method='bfill').fillna(method='ffill')
                
                # Target variable: next period volatility
                target = volatility.shift(-1)
                
                features[symbol] = {
                    'X': feature_df.dropna(),
                    'y': target.dropna()
                }
            except Exception as e:
                st.warning(f"Feature preparation failed for {symbol}: {str(e)}")
                continue
        
        return features
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator with error handling"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Neutral RSI for NaN values
        except:
            return pd.Series(50, index=prices.index)
    
    def train_models(self, features_dict):
        """Train ensemble models with cross-validation"""
        from sklearn.model_selection import cross_val_score
        
        for symbol, data in features_dict.items():
            if len(data['X']) > 100:  # Increased minimum data points for better models
                try:
                    X = data['X'].fillna(0)
                    y = data['y'].fillna(0)
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Enhanced model with better parameters
                    model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                    
                    if np.mean(cv_scores) < -0.1:  # Reasonable performance threshold
                        model.fit(X_scaled, y)
                        self.models[symbol] = model
                        self.scalers[symbol] = scaler
                        self.feature_importance[symbol] = dict(zip(X.columns, model.feature_importances_))
                        
                except Exception as e:
                    st.warning(f"Model training failed for {symbol}: {str(e)}")
    
    def predict_risk(self, price_df, volume_df, horizon=5):
        """Enhanced risk prediction with confidence intervals"""
        predictions = {}
        
        for symbol in price_df.columns:
            if symbol in self.models:
                try:
                    # Prepare current features
                    returns = price_df[symbol].pct_change().dropna()
                    volatility = returns.rolling(window=30).std() * np.sqrt(252)
                    
                    current_features = self._prepare_prediction_features(symbol, price_df, volume_df)
                    
                    if current_features is not None:
                        # Scale and predict
                        X_scaled = self.scalers[symbol].transform(current_features)
                        predicted_volatility = self.models[symbol].predict(X_scaled)[0]
                        
                        # Calculate confidence using standard deviation of trees
                        trees_predictions = [tree.predict(X_scaled)[0] for tree in self.models[symbol].estimators_]
                        confidence = 1 - (np.std(trees_predictions) / (predicted_volatility + 1e-8))
                        
                        predictions[symbol] = {
                            'predicted_volatility': max(predicted_volatility, 0),
                            'risk_score': min(predicted_volatility * 100, 100),
                            'confidence': min(confidence, 1.0),
                            'trend': 'INCREASING' if predicted_volatility > volatility.iloc[-1] else 'DECREASING',
                            'trees_std': np.std(trees_predictions)
                        }
                        
                except Exception as e:
                    st.warning(f"Prediction failed for {symbol}: {str(e)}")
        
        return predictions
    
    def _prepare_prediction_features(self, symbol, price_df, volume_df):
        """Prepare features for prediction"""
        try:
            returns = price_df[symbol].pct_change().dropna()
            volatility = returns.rolling(window=30).std() * np.sqrt(252)
            momentum = price_df[symbol] / price_df[symbol].shift(30) - 1
            rsi = self.calculate_rsi(price_df[symbol])
            volume_ma = volume_df[symbol].rolling(window=30).mean()
            price_ma_ratio = price_df[symbol] / price_df[symbol].rolling(window=50).mean()
            volatility_ratio = volatility / volatility.rolling(window=50).mean()
            
            current_features = pd.DataFrame({
                'volatility': [volatility.iloc[-1] if len(volatility) > 0 else 0],
                'momentum': [momentum.iloc[-1] if len(momentum) > 0 else 0],
                'rsi': [rsi.iloc[-1] if len(rsi) > 0 else 50],
                'volume_ma': [volume_ma.iloc[-1] if len(volume_ma) > 0 else 0],
                'price_ma_ratio': [price_ma_ratio.iloc[-1] if len(price_ma_ratio) > 0 else 1],
                'volatility_ratio': [volatility_ratio.iloc[-1] if len(volatility_ratio) > 0 else 1],
                'returns_lag1': [returns.iloc[-1] if len(returns) > 0 else 0],
                'returns_lag5': [returns.iloc[-5] if len(returns) > 4 else 0],
                'returns_lag10': [returns.iloc[-10] if len(returns) > 9 else 0]
            }).fillna(0)
            
            return current_features
        except:
            return None

class EnterpriseDatabaseManager:
    """Enhanced database management for enterprise use"""
    def __init__(self):
        self.conn = sqlite3.connect('enterprise_risk_manager.db', check_same_thread=False)
        self.init_database()
    
    def init_database(self):
        """Initialize enterprise database tables"""
        # Enhanced Users table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                role TEXT NOT NULL,
                risk_tolerance TEXT,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Enhanced Portfolios table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                portfolio_data TEXT NOT NULL,
                currency TEXT DEFAULT 'KES',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Enhanced Audit log table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                severity TEXT DEFAULT 'INFO',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Risk thresholds table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_thresholds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                threshold_name TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.conn.commit()

class RiskManager:
    """Base Risk Manager Class"""
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.risk_thresholds = {
            'volatility': 0.25,
            'drawdown': 0.15,
            'correlation': 0.8,
            'liquidity': 1000000,
            'var_95': 0.05
        }
    
    def calculate_var_from_positions(self, prices, positions):
        """Calculate Value at Risk from positions"""
        try:
            returns = prices.pct_change().dropna()
            portfolio_returns = pd.Series(0.0, index=returns.index)
            
            for symbol, qty in positions.items():
                if symbol in returns.columns:
                    portfolio_returns += returns[symbol] * qty * prices[symbol].iloc[-1] if len(prices[symbol]) > 0 else 0
            
            var_95 = np.percentile(portfolio_returns.dropna(), 5)
            return abs(var_95)
        except:
            return 0.0
    
    def analyze_volatility_vectorized(self, prices):
        """Analyze volatility for all symbols"""
        try:
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            return volatility
        except:
            return pd.Series(0.0, index=prices.columns)

class EnhancedRiskManager(RiskManager):
    """Enhanced RiskManager with enterprise features"""
    
    def __init__(self, db_manager=None):
        super().__init__(db_manager)
        # Enhanced risk thresholds for enterprise
        self.risk_thresholds = {
            'volatility': 0.20,  # Tighter thresholds
            'drawdown': 0.10,
            'correlation': 0.7,
            'liquidity': 5000000,  # Higher liquidity requirement
            'var_95': 0.03,
            'position_limit': 0.05,  # Tighter position limits
            'concentration_limit': 0.2,
            'daily_loss_limit': 0.01  # Tighter loss limits
        }
    
    def generate_enterprise_report_pdf(self, portfolio_data, risk_metrics, user_info):
        """Generate enterprise-grade PDF report with KES"""
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'ALPHA ENTERPRISE RISK MANAGEMENT REPORT', 0, 1, 'C')
        pdf.ln(5)
        
        # KES Currency Notice
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, 'All amounts displayed in Kenyan Shillings (KES)', 0, 1, 'C')
        pdf.ln(10)
        
        # Enhanced report content...
        return "PDF report generated"

def create_enterprise_dashboard():
    """Enhanced enterprise dashboard"""
    # Initialize enhanced components
    currency_converter = CurrencyConverter()
    ai_explainer = AIRiskExplainer()
    news_integration = RealTimeNewsIntegration()
    alert_system = MobileAlertSystem()
    compliance_dashboard = RegulatoryComplianceDashboard()
    peer_benchmarking = PeerBenchmarking()
    
    # Enhanced session state initialization
    if 'enterprise_initialized' not in st.session_state:
        st.session_state.enterprise_initialized = True
        st.session_state.currency_converter = currency_converter
        st.session_state.ai_explainer = ai_explainer
        st.session_state.news_integration = news_integration
        st.session_state.alert_system = alert_system
        st.session_state.compliance_dashboard = compliance_dashboard
        st.session_state.peer_benchmarking = peer_benchmarking
        st.session_state.kes_mode = True
    
    # Enterprise header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>ALPHA Risk Management System Pro+</h1>
        <h3 style='margin: 0; opacity: 0.9;'>Enterprise Edition | Central Bank of Kenya Compliant</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>Tier-1 Financial Institution Solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Currency toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        kes_mode = st.toggle("🇰🇪 Display in KES", value=True, key="kes_toggle")
        st.session_state.kes_mode = kes_mode
    
    with col3:
        if st.button("🔄 Update Exchange Rates", key="refresh_rates"):
            currency_converter.update_exchange_rates()
            st.success("Exchange rates updated!")
    
    # Display current exchange rate
    if st.session_state.kes_mode:
        st.info(f"💱 Current Exchange Rate: 1 USD = {currency_converter.usd_to_kes:.2f} KES")
    
    # Enhanced executive summary with KES support
    st.markdown("""
    <div class='executive-summary'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h2 style='margin: 0; color: white;'>🎯 Executive Risk Dashboard</h2>
                <p style='margin: 0; opacity: 0.9;'>Real-time Portfolio Monitoring & Risk Analytics</p>
            </div>
            <div style='text-align: right;'>
                <h4 style='margin: 0; color: white;'>🇰🇪 Kenyan Shilling (KES)</h4>
                <p style='margin: 0; opacity: 0.9;'>Enterprise Grade Risk Management</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return currency_converter, ai_explainer, news_integration, alert_system, compliance_dashboard, peer_benchmarking

def enhanced_portfolio_upload_section():
    """Enhanced portfolio upload with KES support"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Enterprise Portfolio Management")
    
    # Portfolio upload with enhanced validation
    uploaded_file = st.sidebar.file_uploader("Upload Portfolio CSV", type=['csv'], 
                                           help="Upload CSV with columns: symbol, quantity, price")
    
    if uploaded_file is not None:
        try:
            user_portfolio = pd.read_csv(uploaded_file)
            
            # Enhanced column normalization
            user_portfolio.columns = user_portfolio.columns.str.lower().str.strip()
            
            # Flexible column mapping
            column_mapping = {
                'ticker': 'symbol', 'sym': 'symbol', 'stock': 'symbol',
                'qty': 'quantity', 'shares': 'quantity', 'units': 'quantity',
                'price': 'price', 'cost': 'price', 'value': 'price'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in user_portfolio.columns:
                    user_portfolio.rename(columns={old_col: new_col}, inplace=True)
            
            # Validate required columns
            required_columns = ['symbol', 'quantity']
            missing_columns = [col for col in required_columns if col not in user_portfolio.columns]
            
            if missing_columns:
                st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.sidebar.info("Required columns: symbol, quantity (price is optional)")
            else:
                # Fill missing prices with current market data
                if 'price' not in user_portfolio.columns:
                    user_portfolio['price'] = np.nan
                
                st.session_state.current_portfolio = user_portfolio.to_dict('records')
                st.sidebar.success("✅ Portfolio uploaded successfully!")
                
                # Display enhanced portfolio preview
                with st.sidebar.expander("Portfolio Preview", expanded=True):
                    display_df = user_portfolio.copy()
                    if st.session_state.kes_mode and 'price' in display_df.columns:
                        display_df['price_kes'] = display_df['price'] * st.session_state.currency_converter.usd_to_kes
                        display_df['price_kes'] = display_df['price_kes'].apply(lambda x: f"KES {x:,.2f}")
                    
                    st.dataframe(display_df.head(10), use_container_width=True)
                    
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            st.sidebar.info("Please ensure the CSV file is properly formatted")

def display_enhanced_risk_metrics(risk_mgr, portfolio_info, positions):
    """Display risk metrics with KES currency support"""
    currency_converter = st.session_state.currency_converter
    kes_mode = st.session_state.kes_mode
    
    # Calculate portfolio value
    portfolio_value = sum(portfolio_info['prices'].iloc[-1][sym] * qty 
                         for sym, qty in positions.items() 
                         if sym in portfolio_info['prices'].columns)
    
    # Calculate VaR
    var_95 = risk_mgr.calculate_var_from_positions(portfolio_info['prices'], positions)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = len(positions)
        st.metric("Total Assets", f"{total_assets}")
    
    with col2:
        if kes_mode:
            display_value = currency_converter.usd_to_kes_format(portfolio_value)
        else:
            display_value = f"${portfolio_value:,.0f}"
        st.metric("Portfolio Value", display_value)
    
    with col3:
        volatility_series = risk_mgr.analyze_volatility_vectorized(portfolio_info['prices'])
        overall_volatility = volatility_series.mean()
        st.metric("Portfolio Volatility", f"{overall_volatility:.2%}")
    
    with col4:
        if kes_mode:
            display_var = currency_converter.usd_to_kes_format(var_95)
        else:
            display_var = f"${var_95:,.0f}"
        st.metric("1-day VaR (95%)", display_var)

def create_ai_risk_explanations_tab(ai_explainer, portfolio_data, risk_metrics):
    """Create AI-powered risk explanations tab"""
    st.subheader("🤖 AI Risk Intelligence")
    
    if not portfolio_data or not risk_metrics:
        st.info("Upload a portfolio to get AI-powered risk analysis")
        return
    
    # Select asset for detailed analysis
    symbols = list(portfolio_data.keys())
    selected_symbol = st.selectbox("Select Asset for AI Analysis", symbols)
    
    if selected_symbol:
        # Get risk metrics for selected symbol
        symbol_metrics = risk_metrics.get(selected_symbol, {})
        
        if symbol_metrics:
            # Generate AI explanation
            explanation = ai_explainer.generate_risk_explanation(
                selected_symbol, 
                symbol_metrics
            )
            
            st.markdown(f"""
            <div class='ai-explanation'>
                <h4 style='color: white; margin: 0;'>AI Risk Analysis: {selected_symbol}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(explanation)
            
            # Show risk factors
            st.subheader("🔍 Key Risk Factors")
            factors = [
                ("Volatility", symbol_metrics.get('volatility', 0), 0.15),
                ("Value at Risk", symbol_metrics.get('var_95', 0), 0.03),
                ("Maximum Drawdown", symbol_metrics.get('max_drawdown', 0), 0.10)
            ]
            
            for factor_name, value, threshold in factors:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{factor_name}**")
                with col2:
                    st.write(f"{value:.2%}")
                with col3:
                    if value > threshold:
                        st.error("⚠️ Above Threshold")
                    else:
                        st.success("✅ Within Limits")

def create_real_time_news_tab(news_integration, symbols=None):
    """Create real-time news integration tab"""
    st.subheader("📰 Market News & Sentiment")
    
    # Get financial news
    news_articles = news_integration.get_financial_news(symbols)
    
    if not news_articles:
        st.info("No recent news available")
        return
    
    # Display news articles with sentiment
    for article in news_articles:
        sentiment = article.get('sentiment', 0)
        impact = article.get('impact', 'MEDIUM')
        
        # Determine sentiment class
        if sentiment > 0.1:
            sentiment_class = "news-positive"
            sentiment_icon = "📈"
        elif sentiment < -0.1:
            sentiment_class = "news-negative" 
            sentiment_icon = "📉"
        else:
            sentiment_class = "news-neutral"
            sentiment_icon = "➡️"
        
        st.markdown(f"""
        <div class='{sentiment_class}'>
            <div style='display: flex; justify-content: between; align-items: start;'>
                <div style='flex: 1;'>
                    <h4 style='margin: 0 0 5px 0;'>{article['title']}</h4>
                    <p style='margin: 0; font-size: 0.9em;'>{article['description']}</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.8em; color: #666;'>
                        {article['source']} • {article['published_at']} • Impact: {impact}
                    </p>
                </div>
                <div style='margin-left: 10px; text-align: center;'>
                    <span style='font-size: 1.5em;'>{sentiment_icon}</span>
                    <br>
                    <span style='font-size: 0.8em;'>Sentiment: {sentiment:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_compliance_dashboard_tab(compliance_dashboard, portfolio_metrics):
    """Create regulatory compliance dashboard tab"""
    st.subheader("🏛️ CBK Regulatory Compliance")
    
    # Simulated institution data
    institution_data = {
        'capital_adequacy_ratio': 0.125,  # 12.5%
        'liquidity_coverage_ratio': 1.15,  # 115%
        'last_stress_test': datetime.now() - timedelta(days=45)
    }
    
    # Check compliance status
    compliance_status = compliance_dashboard.check_compliance_status(
        portfolio_metrics, 
        institution_data
    )
    
    # Generate compliance report
    compliance_report = compliance_dashboard.generate_compliance_report(compliance_status)
    
    # Display overall compliance status
    overall_status = "COMPLIANT" if compliance_report['overall_compliant'] else "NON-COMPLIANT"
    status_color = "compliance-pass" if compliance_report['overall_compliant'] else "compliance-fail"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 10px 0;'>
        <h3 style='margin: 0;'>Overall Compliance Status</h3>
        <div class='{status_color}' style='display: inline-block; margin: 10px 0; padding: 10px 20px;'>
            {overall_status}
        </div>
        <p style='margin: 5px 0;'>Next Reporting Date: {compliance_report['next_reporting_date']}</p>
        <p style='margin: 5px 0; font-size: 0.9em;'>Regulatory Contact: {compliance_report['regulatory_contact']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display detailed compliance metrics
    st.subheader("Compliance Metrics")
    
    for metric_name, status in compliance_status.items():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{status['description']}**")
        
        with col2:
            st.write(f"Required: {status['required']}")
        
        with col3:
            st.write(f"Actual: {status['actual']}")
        
        with col4:
            if status['compliant']:
                st.success("✅ COMPLIANT")
            else:
                st.error("❌ NON-COMPLIANT")

def create_mobile_alerts_tab(alert_system, portfolio_metrics):
    """Create mobile alerts configuration tab"""
    st.subheader("📱 Enterprise Alert System")
    
    # Alert preferences
    st.subheader("Alert Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_threshold = st.slider(
            "VaR Alert Threshold (%)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        volatility_threshold = st.slider(
            "Volatility Alert Threshold (%)", 
            min_value=10.0,
            max_value=50.0,
            value=20.0,
            step=1.0
        )
    
    with col2:
        concentration_threshold = st.slider(
            "Concentration Alert Threshold (%)",
            min_value=10.0,
            max_value=50.0, 
            value=30.0,
            step=1.0
        )
        
        # Notification methods
        sms_alerts = st.checkbox("Enable SMS Alerts", value=True)
        email_alerts = st.checkbox("Enable Email Alerts", value=True)
    
    # Contact information
    st.subheader("Contact Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        phone_number = st.text_input("Mobile Number for SMS Alerts", value="+254700000000")
    
    with col2:
        email_address = st.text_input("Email for Alert Notifications", value="admin@company.co.ke")
    
    # Test alerts
    if st.button("🧪 Test Alert System", type="secondary"):
        user_preferences = {
            'var_threshold': var_threshold / 100,
            'volatility_threshold': volatility_threshold / 100, 
            'concentration_threshold': concentration_threshold / 100
        }
        
        # Check for threshold breaches
        alerts = alert_system.check_risk_thresholds(portfolio_metrics, user_preferences)
        
        if alerts:
            for alert in alerts:
                st.warning(f"🚨 {alert['message']}")
                
                # Send test alerts
                if sms_alerts and phone_number:
                    alert_system.send_sms_alert(phone_number, f"TEST: {alert['message']}")
                
                if email_alerts and email_address:
                    alert_system.send_email_alert(
                        email_address, 
                        f"TEST ALERT: {alert['type']}",
                        alert['message']
                    )
        else:
            st.success("✅ No threshold breaches detected")
    
    # Current portfolio metrics vs thresholds
    st.subheader("Current Metrics vs Thresholds")
    
    metrics_comparison = [
        ("Value at Risk", portfolio_metrics.get('var_95', 0), var_threshold / 100),
        ("Volatility", portfolio_metrics.get('volatility', 0), volatility_threshold / 100),
        ("Concentration", portfolio_metrics.get('concentration', 0), concentration_threshold / 100)
    ]
    
    for metric_name, current_value, threshold in metrics_comparison:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{metric_name}**")
        
        with col2:
            st.write(f"{current_value:.2%}")
        
        with col3:
            st.write(f"{threshold:.2%}")
        
        with col4:
            if current_value > threshold:
                st.error("🔴 Above Threshold")
            else:
                st.success("🟢 Within Limits")

def create_peer_benchmarking_tab(peer_benchmarking, portfolio_metrics):
    """Create peer benchmarking tab"""
    st.subheader("📊 Peer Institution Benchmarking")
    
    # Select institution type for comparison
    institution_type = st.selectbox(
        "Select Your Institution Type",
        ["tier_1_banks", "insurance", "saccos", "microfinance"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Get peer comparison
    comparison = peer_benchmarking.get_peer_comparison(portfolio_metrics, institution_type)
    
    if not comparison:
        st.info("No benchmarking data available for selected institution type")
        return
    
    # Display benchmarking results
    st.subheader("Performance vs Peer Average")
    
    for metric, data in comparison.items():
        if metric not in ['top_performer', 'peer_group_size']:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{metric.replace('_', ' ').title()}**")
            
            with col2:
                st.write(f"{data['your_value']:.2%}")
            
            with col3:
                st.write(f"{data['peer_average']:.2%}")
            
            with col4:
                outperformance = data['outperformance']
                if outperformance > 0:
                    st.success(f"↑ +{outperformance:.2%}")
                else:
                    st.error(f"↓ {outperformance:.2%}")
    
    # Overall percentile ranking
    avg_percentile = np.mean([data['percentile'] for metric, data in comparison.items() 
                             if metric not in ['top_performer', 'peer_group_size']])
    
    st.subheader("Overall Ranking")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Your Percentile Rank", f"{avg_percentile:.0f}th")
    
    with col2:
        st.metric("Top Performer", comparison.get('top_performer', 'Unknown'))
    
    with col3:
        st.metric("Peer Group Size", comparison.get('peer_group_size', 0))
    
    # Performance visualization
    st.subheader("Performance Distribution")
    
    # Simulated peer distribution
    peer_data = np.random.normal(0.1, 0.05, 1000)  # Simulated peer returns
    
    fig = go.Figure()
    
    # Add histogram of peer performance
    fig.add_trace(go.Histogram(
        x=peer_data,
        name='Peer Institutions',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Add vertical line for user's performance
    user_performance = portfolio_metrics.get('return', 0.12)  # Default if not available
    fig.add_vline(
        x=user_performance,
        line_dash="dash",
        line_color="red",
        annotation_text="Your Performance"
    )
    
    fig.update_layout(
        title="Performance Distribution vs Peers",
        xaxis_title="Returns",
        yaxis_title="Number of Institutions",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Enhanced main function with all new features"""
    try:
        # Initialize enhanced dashboard
        currency_converter, ai_explainer, news_integration, alert_system, compliance_dashboard, peer_benchmarking = create_enterprise_dashboard()
        enhanced_portfolio_upload_section()
        
        # Initialize enhanced components
        db_manager = EnterpriseDatabaseManager()
        risk_mgr = EnhancedRiskManager(db_manager)
        
        # Create main tabs with new features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "🤖 AI Analysis", 
            "📰 Market News", 
            "🏛️ Compliance",
            "📱 Alerts", 
            "📊 Benchmarking"
        ])
        
        # Sample data for demonstration
        sample_portfolio = {
            'SCOM': {'quantity': 1000, 'price': 2.5},
            'KCB': {'quantity': 500, 'price': 4.0},
            'EQTY': {'quantity': 800, 'price': 3.2}
        }
        
        sample_risk_metrics = {
            'SCOM': {'volatility': 0.22, 'var_95': 0.028, 'max_drawdown': 0.15},
            'KCB': {'volatility': 0.18, 'var_95': 0.022, 'max_drawdown': 0.12},
            'EQTY': {'volatility': 0.25, 'var_95': 0.035, 'max_drawdown': 0.18}
        }
        
        portfolio_metrics = {
            'var_95': 0.028,
            'volatility': 0.21,
            'concentration': 0.35,
            'return': 0.14
        }
        
        with tab1:
            st.subheader("Portfolio Overview")
            # Display enhanced risk metrics
            if 'current_portfolio' in st.session_state:
                # Convert portfolio to positions format
                positions = {item['symbol']: item['quantity'] for item in st.session_state.current_portfolio}
                
                # Create sample price data for demonstration
                dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
                price_data = {}
                for symbol in positions.keys():
                    # Generate realistic price data
                    prices = [100]
                    for i in range(1, len(dates)):
                        returns = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
                        prices.append(prices[-1] * (1 + returns))
                    price_data[symbol] = prices
                
                price_df = pd.DataFrame(price_data, index=dates)
                volume_df = pd.DataFrame({sym: np.random.randint(10000, 1000000, len(dates)) for sym in positions.keys()}, index=dates)
                
                portfolio_info = {
                    'prices': price_df,
                    'volumes': volume_df
                }
                
                display_enhanced_risk_metrics(risk_mgr, portfolio_info, positions)
            else:
                st.info("Please upload a portfolio to view risk metrics")
        
        with tab2:
            create_ai_risk_explanations_tab(ai_explainer, sample_portfolio, sample_risk_metrics)
        
        with tab3:
            create_real_time_news_tab(news_integration, list(sample_portfolio.keys()))
        
        with tab4:
            create_compliance_dashboard_tab(compliance_dashboard, portfolio_metrics)
        
        with tab5:
            create_mobile_alerts_tab(alert_system, portfolio_metrics)
        
        with tab6:
            create_peer_benchmarking_tab(peer_benchmarking, portfolio_metrics)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div class='copyright'>
            <p>ALPHA Risk Manager Pro+ Enterprise Edition v3.0 | © 2024 Alpha Financial Technologies</p>
            <p>Central Bank of Kenya Compliant | Tier-1 Financial Institution Solution</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Enterprise system error: {str(e)}")
        st.info("Please refresh the application or contact system administrator")

if __name__ == "__main__":
    main()