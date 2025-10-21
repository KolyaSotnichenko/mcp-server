from fastmcp import FastMCP
import requests
import os
import subprocess
import tempfile
import shutil
import threading
import time
import json
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("My MCP Server")

# Global dictionary to track temporary directories
temp_dirs = {}

def cleanup_expired_dirs():
    """Clean up expired temporary directories"""
    global temp_dirs
    current_time = datetime.now()
    dirs_to_remove = []
    
    for temp_dir, expiry_time in temp_dirs.items():
        if current_time > expiry_time and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                dirs_to_remove.append(temp_dir)
                print(f"Cleaned up expired directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up {temp_dir}: {e}")
    
    for temp_dir in dirs_to_remove:
        del temp_dirs[temp_dir]

def schedule_cleanup():
    """Schedule periodic cleanup"""
    while True:
        cleanup_expired_dirs()
        time.sleep(300)

# Start cleanup thread
cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
cleanup_thread.start()

@mcp.tool(
    name="get_harry_potter_character",
    description="Gets information about a Harry Potter character by name",
)
def get_harry_potter_character(character_name: str = None) -> str:
    """Gets information about a Harry Potter character by name"""

    if character_name is None:
        return "Please provide a character name."

    url = "https://hp-api.onrender.com/api/characters"
    response = requests.get(url)
    characters = response.json()
    if response.status_code != 200:
        return "Error fetching data from API."
    for char in characters:
        if char['name'].lower() == character_name.lower():
            return f"Name: {char['name']}\nSpecies: {char['species']}\nHouse: {char.get('house', 'Unknown')}\nActor: {char.get('actor', 'Unknown')}\nWand: {char.get('wand', {}).get('wood', 'Unknown')} {char.get('wand', {}).get('core', '')} {char.get('wand', {}).get('length', '')}"
    return f"Character '{character_name}' not found."

@mcp.tool(
    name="query_perplexity",
    description="Makes a query to Perplexity AI for real-time search and information",
)
def query_perplexity(query: str) -> str:
    """Makes a query to Perplexity AI for real-time search and information"""

    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not set in environment variables."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return f"Error: API request failed with status {response.status_code} and message: {response.text}"

    result = response.json()
    return result['choices'][0]['message']['content']

@mcp.tool(
    name="crypto_analyzer_advanced",
    description="Advanced cryptocurrency analysis using TAAPI.IO with 200+ technical indicators",
)
def crypto_analyzer_advanced(crypto_symbol: str, timeframe: str = "1d", exchange: str = "binance") -> str:
    """Advanced cryptocurrency analysis with professional technical indicators.

    Args:
        crypto_symbol: Cryptocurrency symbol (e.g., BTC/USDT, ETH/USDT)
        timeframe: Analysis timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        exchange: Exchange name (binance, coinbase, kraken, etc.)

    Returns:
        str: JSON with comprehensive trading analysis, SL/TP levels, and technical indicators
    """
    try:
        # Get TAAPI.IO API key from environment
        taapi_key = os.environ.get('TAAPI_API_KEY')
        if not taapi_key:
            return json.dumps({
                "error": "TAAPI_API_KEY not set in environment variables. Get free key at https://taapi.io",
                "symbol": crypto_symbol,
                "status": "error"
            }, indent=2)
        
        # Normalize symbol format
        if '/' not in crypto_symbol:
            symbol = f"{crypto_symbol.upper()}/USDT"
        else:
            symbol = crypto_symbol.upper()
        
        base_url = "https://api.taapi.io"
        
        # Get multiple technical indicators in one request
        indicators = {
            "rsi": f"{base_url}/rsi",
            "macd": f"{base_url}/macd", 
            "bb": f"{base_url}/bbands",
            "sma20": f"{base_url}/sma",
            "sma50": f"{base_url}/sma",
            "ema12": f"{base_url}/ema",
            "ema26": f"{base_url}/ema",
            "stoch": f"{base_url}/stoch",
            "adx": f"{base_url}/adx",
            "cci": f"{base_url}/cci",
            "williams": f"{base_url}/willr",
            "atr": f"{base_url}/atr"
        }
        
        results = {}
        
        # Fetch RSI
        rsi_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe,
            "period": 14
        }
        rsi_response = requests.get(indicators["rsi"], params=rsi_params)
        if rsi_response.status_code == 200:
            results["rsi"] = rsi_response.json()
        
        # Fetch MACD
        macd_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe
        }
        macd_response = requests.get(indicators["macd"], params=macd_params)
        if macd_response.status_code == 200:
            results["macd"] = macd_response.json()
        
        # Fetch Bollinger Bands
        bb_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe,
            "period": 20
        }
        bb_response = requests.get(indicators["bb"], params=bb_params)
        if bb_response.status_code == 200:
            results["bollinger"] = bb_response.json()
        
        # Fetch SMA 20
        sma20_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe,
            "period": 20
        }
        sma20_response = requests.get(indicators["sma20"], params=sma20_params)
        if sma20_response.status_code == 200:
            results["sma20"] = sma20_response.json()
        
        # Fetch SMA 50
        sma50_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe,
            "period": 50
        }
        sma50_response = requests.get(indicators["sma50"], params=sma50_params)
        if sma50_response.status_code == 200:
            results["sma50"] = sma50_response.json()
        
        # Fetch Stochastic
        stoch_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe
        }
        stoch_response = requests.get(indicators["stoch"], params=stoch_params)
        if stoch_response.status_code == 200:
            results["stochastic"] = stoch_response.json()
        
        # Fetch ADX
        adx_params = {
            "secret": taapi_key,
            "exchange": exchange,
            "symbol": symbol,
            "interval": timeframe,
            "period": 14
        }
        adx_response = requests.get(indicators["adx"], params=adx_params)
        if adx_response.status_code == 200:
            results["adx"] = adx_response.json()
        
        # Get current price from CoinGecko for additional context
        coingecko_id_map = {
            'BTC/USDT': 'bitcoin',
            'ETH/USDT': 'ethereum',
            'ADA/USDT': 'cardano',
            'BNB/USDT': 'binancecoin',
            'XRP/USDT': 'ripple',
            'SOL/USDT': 'solana',
            'DOGE/USDT': 'dogecoin',
            'DOT/USDT': 'polkadot',
            'MATIC/USDT': 'matic-network',
            'AVAX/USDT': 'avalanche-2'
        }
        
        coin_id = coingecko_id_map.get(symbol, symbol.split('/')[0].lower())
        
        price_url = f"https://api.coingecko.com/api/v3/simple/price"
        price_params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        price_response = requests.get(price_url, params=price_params)
        current_price = 0
        change_24h = 0
        volume_24h = 0
        
        if price_response.status_code == 200:
            price_data = price_response.json()
            if coin_id in price_data:
                current_price = price_data[coin_id]['usd']
                change_24h = price_data[coin_id].get('usd_24h_change', 0)
                volume_24h = price_data[coin_id].get('usd_24h_vol', 0)
        
        # Advanced technical analysis
        def analyze_signals(indicators_data):
            signals = []
            score = 0
            
            # RSI Analysis
            if "rsi" in indicators_data and "value" in indicators_data["rsi"]:
                rsi_value = indicators_data["rsi"]["value"]
                if rsi_value > 70:
                    signals.append("RSI: Overbought (Sell signal)")
                    score -= 2
                elif rsi_value < 30:
                    signals.append("RSI: Oversold (Buy signal)")
                    score += 2
                elif 40 <= rsi_value <= 60:
                    signals.append("RSI: Neutral")
                else:
                    signals.append(f"RSI: {rsi_value:.1f}")
            
            # MACD Analysis
            if "macd" in indicators_data:
                macd_data = indicators_data["macd"]
                if "valueMACD" in macd_data and "valueMACDSignal" in macd_data:
                    macd_line = macd_data["valueMACD"]
                    signal_line = macd_data["valueMACDSignal"]
                    if macd_line > signal_line:
                        signals.append("MACD: Bullish crossover")
                        score += 1
                    else:
                        signals.append("MACD: Bearish crossover")
                        score -= 1
            
            # Bollinger Bands Analysis
            if "bollinger" in indicators_data:
                bb_data = indicators_data["bollinger"]
                if all(k in bb_data for k in ["valueLowerBand", "valueUpperBand", "valueMiddleBand"]):
                    if current_price > bb_data["valueUpperBand"]:
                        signals.append("BB: Price above upper band (Overbought)")
                        score -= 1
                    elif current_price < bb_data["valueLowerBand"]:
                        signals.append("BB: Price below lower band (Oversold)")
                        score += 1
                    else:
                        signals.append("BB: Price within bands")
            
            # Moving Averages Analysis
            if "sma20" in indicators_data and "sma50" in indicators_data:
                sma20 = indicators_data["sma20"].get("value", 0)
                sma50 = indicators_data["sma50"].get("value", 0)
                if sma20 > sma50:
                    signals.append("SMA: Golden cross (Bullish)")
                    score += 1
                else:
                    signals.append("SMA: Death cross (Bearish)")
                    score -= 1
            
            # Stochastic Analysis
            if "stochastic" in indicators_data:
                stoch_data = indicators_data["stochastic"]
                if "valueK" in stoch_data:
                    stoch_k = stoch_data["valueK"]
                    if stoch_k > 80:
                        signals.append("Stochastic: Overbought")
                        score -= 1
                    elif stoch_k < 20:
                        signals.append("Stochastic: Oversold")
                        score += 1
            
            # ADX Analysis (Trend Strength)
            if "adx" in indicators_data and "value" in indicators_data["adx"]:
                adx_value = indicators_data["adx"]["value"]
                if adx_value > 25:
                    signals.append(f"ADX: Strong trend ({adx_value:.1f})")
                else:
                    signals.append(f"ADX: Weak trend ({adx_value:.1f})")
            
            return signals, score
        
        signals, total_score = analyze_signals(results)
        
        # Generate recommendation based on score
        if total_score >= 3:
            recommendation = "STRONG BUY"
            confidence = "High"
            sl_percentage = 2
            tp_percentage = 8
        elif total_score >= 1:
            recommendation = "BUY"
            confidence = "Medium"
            sl_percentage = 3
            tp_percentage = 6
        elif total_score <= -3:
            recommendation = "STRONG SELL"
            confidence = "High"
            sl_percentage = 2
            tp_percentage = 8
        elif total_score <= -1:
            recommendation = "SELL"
            confidence = "Medium"
            sl_percentage = 3
            tp_percentage = 6
        else:
            recommendation = "HOLD"
            confidence = "Low"
            sl_percentage = 4
            tp_percentage = 4
        
        # Calculate SL and TP levels
        if recommendation in ["STRONG BUY", "BUY"]:
            stop_loss = current_price * (1 - sl_percentage / 100)
            take_profit = current_price * (1 + tp_percentage / 100)
        elif recommendation in ["STRONG SELL", "SELL"]:
            stop_loss = current_price * (1 + sl_percentage / 100)
            take_profit = current_price * (1 - tp_percentage / 100)
        else:  # HOLD
            stop_loss = current_price * (1 - sl_percentage / 100)
            take_profit = current_price * (1 + tp_percentage / 100)
        
        # Prepare comprehensive analysis
        analysis = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "market_data": {
                "current_price": round(current_price, 6) if current_price else "N/A",
                "price_change_24h": round(change_24h, 2) if change_24h else "N/A",
                "volume_24h": volume_24h if volume_24h else "N/A"
            },
            "technical_indicators": results,
            "signals_analysis": signals,
            "recommendation": {
                "action": recommendation,
                "confidence": confidence,
                "score": total_score,
                "max_score": 10
            },
            "trading_levels": {
                "current_price": round(current_price, 6) if current_price else "N/A",
                "stop_loss": round(stop_loss, 6) if current_price else "N/A",
                "take_profit": round(take_profit, 6) if current_price else "N/A",
                "sl_percentage": sl_percentage,
                "tp_percentage": tp_percentage
            },
            "risk_management": {
                "position_size": "1-2% of portfolio",
                "risk_reward_ratio": f"1:{tp_percentage/sl_percentage:.1f}",
                "max_loss": f"{sl_percentage}%"
            },
            "disclaimer": "This analysis is based on technical indicators and is not financial advice. Always do your own research and consider your risk tolerance."
        }
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Advanced analysis failed: {str(e)}",
            "symbol": crypto_symbol,
            "status": "error",
            "suggestion": "Check TAAPI_API_KEY and symbol format. Get free API key at https://taapi.io"
        }, indent=2)

@mcp.tool(
    name="code_executor",
    description="Execute code and automatically return any created files to LLM",
)
def code_executor(source_code: str, language: str = "python") -> str:
    """Execute code in a temporary directory and return results with all created files."""
    global temp_dirs
    
    try:
        # Create temporary directory for this execution
        temp_dir = tempfile.mkdtemp(prefix="mcp_code_exec_")
        expiry_time = datetime.now() + timedelta(hours=1)
        temp_dirs[temp_dir] = expiry_time
        
        # Save current directory
        original_cwd = os.getcwd()
        
        try:
            # Change to temporary directory
            os.chdir(temp_dir)
            
            # Create temporary file with the code
            if language.lower() == "python":
                code_file = os.path.join(temp_dir, "temp_code.py")
                interpreter = "python"
            elif language.lower() in ["javascript", "js"]:
                code_file = os.path.join(temp_dir, "temp_code.js")
                interpreter = "node"
            elif language.lower() in ["bash", "sh"]:
                code_file = os.path.join(temp_dir, "temp_code.sh")
                interpreter = "bash"
            else:
                return json.dumps({
                    "error": f"Unsupported language: {language}",
                    "supported_languages": ["python", "javascript", "bash"]
                }, indent=2)
            
            # Write code to file
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(source_code)
            
            # Execute the code
            try:
                result = subprocess.run(
                    [interpreter, code_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=temp_dir
                )
                
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode
                
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "Execution timed out after 30 seconds"
                exit_code = 124
            except FileNotFoundError:
                stdout = ""
                stderr = f"Interpreter '{interpreter}' not found. Please install {language}."
                exit_code = 127
            
            # Collect all created files for LLM
            files_for_llm = {}
            
            for filename in os.listdir(temp_dir):
                if filename == os.path.basename(code_file):
                    continue
                    
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            
                        try:
                            text_content = file_content.decode('utf-8')
                            files_for_llm[filename] = {
                                "type": "text",
                                "content": text_content,
                                "base64": base64.b64encode(file_content).decode('utf-8'),
                                "size": len(file_content),
                                "encoding": "utf-8",
                                "downloadable": True
                            }
                        except UnicodeDecodeError:
                            files_for_llm[filename] = {
                                "type": "binary",
                                "content": f"Binary file ({len(file_content)} bytes) - use base64 for download",
                                "base64": base64.b64encode(file_content).decode('utf-8'),
                                "size": len(file_content),
                                "encoding": "binary",
                                "downloadable": True
                            }
                    except Exception as e:
                        files_for_llm[filename] = {
                            "type": "error",
                            "content": f"Error reading file: {str(e)}",
                            "base64": "",
                            "size": 0,
                            "encoding": "error",
                            "downloadable": False
                        }
            
            result_data = {
                "execution_status": "completed",
                "language": language,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "files_created": files_for_llm,
                "files_count": len(files_for_llm),
                "temp_directory": temp_dir,
                "expiry_time": expiry_time.isoformat(),
                "message": f"Execution completed. {len(files_for_llm)} files created and ready for download."
            }
            
            return json.dumps(result_data, indent=2)
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        return json.dumps({
            "execution_status": "failed",
            "error": f"Execution failed: {str(e)}",
            "language": language,
            "exit_code": 1,
            "files_created": {},
            "files_count": 0
        }, indent=2)

if __name__ == "__main__":
    mcp.run(
        transport="http", 
        host="0.0.0.0",
        port=8000,
        path="/",
        log_level="debug",
    )