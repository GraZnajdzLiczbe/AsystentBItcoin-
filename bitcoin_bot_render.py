# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# --- Bot Gie³dowy v6.4 (dla Render) ---
# ---------------------------------------------------------------------------
#
# OSTRZE¯ENIE:
# Ten skrypt jest przeznaczony WY£¥CZNIE do celów edukacyjnych.
#
# NOWOŒCI:
# - Czysta wersja zoptymalizowana do dzia³ania jako "Background Worker" na Render.
# - Domyœlny stan to "IN_POSITION".
# ---------------------------------------------------------------------------

import configparser
import time
import requests
import pandas as pd
import pandas_ta as ta
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# --- Konfiguracja Logowania ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- G£ÓWNA KONFIGURACJA BOTA ---
COIN_ID = 'bitcoin'
VS_CURRENCY = 'usdc'
DAYS_OF_DATA = 90
CHECK_INTERVAL_SECONDS = 3600  # 1 godzina
SCORE_THRESHOLD = 3

# --- PARAMETRY WSKANIKÓW ---
RSI_PERIOD = 14
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
STOCH_K, STOCH_D, STOCH_SMOOTH_K = 14, 3, 3
STOCH_OVERBOUGHT, STOCH_OVERSOLD = 80, 20
BBANDS_PERIOD, BBANDS_STD = 20, 2.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_FACTOR = 1.75

# --- PARAMETRY SENTYMENTU ---
NEWS_KEYWORDS = "bitcoin"
SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD = 0.3, -0.3

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except ImportError:
    print("VADER Sentiment library not found. News analysis will be disabled.")
    analyzer = None

# --- Funkcje analityczne ---
def get_news_sentiment(api_key: str) -> (float, int):
    if not analyzer: return 0.0, 0
    url = f"https://gnews.io/api/v4/search?q={NEWS_KEYWORDS}&lang=en&max=10&token={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        if not articles: return 0.0, 0
        total_score = sum(analyzer.polarity_scores(a.get('title', ''))['compound'] for a in articles)
        return total_score / len(articles), len(articles)
    except requests.exceptions.RequestException as e:
        logging.error(f"B³¹d podczas pobierania wiadomoœci: {e}")
        return 0.0, 0

def get_market_data(coin_id, vs_currency, days):
    try:
        ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params_ohlc = {'vs_currency': vs_currency, 'days': days}
        response_ohlc = requests.get(ohlc_url, params=params_ohlc)
        response_ohlc.raise_for_status()
        df_ohlc = pd.DataFrame(response_ohlc.json(), columns=['time', 'open', 'high', 'low', 'close'])
        df_ohlc['time'] = pd.to_datetime(df_ohlc['time'], unit='ms').dt.date

        chart_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params_chart = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
        response_chart = requests.get(chart_url, params=params_chart)
        response_chart.raise_for_status()
        df_volume = pd.DataFrame(response_chart.json()['total_volumes'], columns=['time', 'volume'])
        df_volume['time'] = pd.to_datetime(df_volume['time'], unit='ms').dt.date

        df = pd.merge(df_ohlc, df_volume, on='time')
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df[~df.index.duplicated(keep='first')].sort_index()
    except Exception as e:
        logging.error(f"B³¹d podczas pobierania danych z CoinGecko: {e}")
        return None

def analyze_market_state(news_api_key: str):
    logging.info("Rozpoczynanie nowego cyklu analitycznego...")
    market_data = get_market_data(COIN_ID, VS_CURRENCY, DAYS_OF_DATA)
    if market_data is None or len(market_data) < MACD_SLOW:
        return 'HOLD', {}
    
    market_data.ta.rsi(length=RSI_PERIOD, append=True)
    market_data.ta.stoch(k=STOCH_K, d=STOCH_D, smooth_k=STOCH_SMOOTH_K, append=True)
    market_data.ta.bbands(length=BBANDS_PERIOD, std=BBANDS_STD, append=True)
    market_data.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    market_data['volume_ma'] = market_data['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    market_data.dropna(inplace=True)
    
    last, previous = market_data.iloc[-1], market_data.iloc[-2]
    buy_score, sell_score = 0, 0
    buy_reasons, sell_reasons = [], []
    
    if last[f'RSI_{RSI_PERIOD}'] < RSI_OVERSOLD: buy_score += 1; buy_reasons.append("RSI < 30")
    if last[f'RSI_{RSI_PERIOD}'] > RSI_OVERBOUGHT: sell_score += 1; sell_reasons.append("RSI > 70")
    stoch_k_col = f'STOCHk_{STOCH_K}_{STOCH_D}_{STOCH_SMOOTH_K}'
    if last[stoch_k_col] < STOCH_OVERSOLD: buy_score += 1; buy_reasons.append("Stochastic < 20")
    if last[stoch_k_col] > STOCH_OVERBOUGHT: sell_score += 1; sell_reasons.append("Stochastic > 80")
    if last['close'] < last[f'BBL_{BBANDS_PERIOD}_{BBANDS_STD}']: buy_score += 1; buy_reasons.append("Cena pod wstêg¹ Bollingera")
    if last['close'] > last[f'BBU_{BBANDS_PERIOD}_{BBANDS_STD}']: sell_score += 1; sell_reasons.append("Cena nad wstêg¹ Bollingera")
    macd_line = f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'; signal_line = f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    if last[macd_line] > last[signal_line] and previous[macd_line] <= previous[signal_line]: buy_score += 1; buy_reasons.append("Z³oty Krzy¿ MACD")
    if last[macd_line] < last[signal_line] and previous[macd_line] >= previous[signal_line]: sell_score += 1; sell_reasons.append("Krzy¿ Œmierci MACD")
    
    is_volume_spike = last['volume'] > last['volume_ma'] * VOLUME_SPIKE_FACTOR
    volume_analysis_text = f"Zwyk³y ({last['volume'] / last['volume_ma']:.1f}x œr.)"
    if is_volume_spike:
        volume_analysis_text = f"WYSOKI! ({last['volume'] / last['volume_ma']:.1f}x œr.)"
        if last['close'] > previous['close']: buy_score += 1; buy_reasons.append("Wzrost ceny przy wysokim wolumenie")
        elif last['close'] < previous['close']: sell_score += 1; sell_reasons.append("Spadek ceny przy wysokim wolumenie")

    avg_sentiment, articles_found = get_news_sentiment(news_api_key)
    if articles_found > 0:
        if avg_sentiment > SENTIMENT_POSITIVE_THRESHOLD: buy_score += 1; buy_reasons.append(f"Pozytywny sentyment ({avg_sentiment:.2f})")
        if avg_sentiment < SENTIMENT_NEGATIVE_THRESHOLD: sell_score += 1; sell_reasons.append(f"Negatywny sentyment ({avg_sentiment:.2f})")
    
    analysis_data = {"cena": f"{last['close']:,.2f} {VS_CURRENCY.upper()}", "rsi": f"{last[f'RSI_{RSI_PERIOD}']:.2f}", "stoch": f"{last[stoch_k_col]:.2f}", "sentiment": f"{avg_sentiment:.2f}", "volume": volume_analysis_text}
    
    if buy_score >= SCORE_THRESHOLD:
        analysis_data["reasons"] = ", ".join(buy_reasons)
        analysis_data["scores"] = f"KUP: {buy_score} | SPRZEDAJ: {sell_score}"
        return 'BUY', analysis_data
    if sell_score >= SCORE_THRESHOLD:
        analysis_data["reasons"] = ", ".join(sell_reasons)
        analysis_data["scores"] = f"KUP: {buy_score} | SPRZEDAJ: {sell_score}"
        return 'SELL', analysis_data
        
    logging.info(f"Zakoñczono analizê. Sygna³: HOLD (Kup: {buy_score}, Sprzedaj: {sell_score})")
    return 'HOLD', {}

# --- Funkcje obs³ugi komend Telegrama ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.bot_data.setdefault('state', 'IN_POSITION')
    context.bot_data.setdefault('last_signal', None)
    instructions = ("?? *Witaj w Bocie v6.4 (dla Render)!*\n\nBot uruchomiony w stanie 'W POZYCJI'. Szuka sygna³u do SPRZEDA¯Y.\n\n/status - Sprawdza stan.\n/potwierdzam - PotwierdŸ akcjê.\n/ignoruje - Zignoruj sygna³.\n/reset - Resetuj stan.")
    await update.message.reply_text(instructions, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.bot_data.get('state', 'NEUTRAL')
    status_text = f"?? *Aktualny stan bota:* `{state}`\n"
    if state == 'NEUTRAL': status_text += "Szukam sygna³u KUP."
    elif state == 'IN_POSITION': status_text += "Szukam sygna³u SPRZEDAJ."
    elif state == 'AWAITING_CONFIRMATION': status_text += f"Wys³a³em sygna³ *{context.bot_data.get('last_signal')}*. Czekam na reakcjê."
    await update.message.reply_text(status_text, parse_mode='Markdown')

async def confirm_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data.get('state') != 'AWAITING_CONFIRMATION':
        await update.message.reply_text("Brak akcji do potwierdzenia.")
        return
    if context.bot_data.get('last_signal') == 'BUY':
        context.bot_data['state'] = 'IN_POSITION'
        await update.message.reply_text("? Potwierdzone! Stan: 'W POZYCJI'.")
    elif context.bot_data.get('last_signal') == 'SELL':
        context.bot_data['state'] = 'NEUTRAL'
        await update.message.reply_text("? Potwierdzone! Stan: 'NEUTRALNY'.")
    context.bot_data['last_signal'] = None

async def ignore_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data.get('state') != 'AWAITING_CONFIRMATION':
        await update.message.reply_text("Brak sygna³u do zignorowania.")
        return
    if context.bot_data.get('last_signal') == 'SELL': context.bot_data['state'] = 'IN_POSITION'
    else: context.bot_data['state'] = 'NEUTRAL'
    context.bot_data['last_signal'] = None
    await update.message.reply_text(f"?? Sygna³ zignorowany. Wracam do stanu '{context.bot_data['state']}'.")

async def reset_state(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.bot_data['state'] = 'NEUTRAL'
    context.bot_data['last_signal'] = None
    await update.message.reply_text("?? Stan bota zresetowany do 'NEUTRALNY'.")

async def analysis_job(context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data.get('state') == 'AWAITING_CONFIRMATION':
        return
    market_signal, data = analyze_market_state(context.job.data['news_api_key'])
    signal_to_send = None
    if context.bot_data.get('state') == 'NEUTRAL' and market_signal == 'BUY': signal_to_send = 'BUY'
    elif context.bot_data.get('state') == 'IN_POSITION' and market_signal == 'SELL': signal_to_send = 'SELL'
    if signal_to_send:
        notification_text = (f"{'??' if signal_to_send == 'BUY' else '??'} *Sygna³ {signal_to_send} dla {COIN_ID.capitalize()}!*\n\n"
                             f"*{data.get('scores', '')}*\n*Powody:* `{data.get('reasons', 'Brak')}`\n\n"
                             f"*Cena:* `{data.get('cena', 'N/A')}`\n"
                             f"*Wolumen:* `{data.get('volume', 'N/A')}`\n"
                             f"*Sentyment:* `{data.get('sentiment', 'N/A')}`\n\n"
                             f"?? Reaguj: /potwierdzam lub /ignoruje")
        await context.bot.send_message(chat_id=context.job.chat_id, text=notification_text, parse_mode='Markdown')
        context.bot_data['state'] = 'AWAITING_CONFIRMATION'
        context.bot_data['last_signal'] = signal_to_send

# --- G³ówna funkcja uruchomieniowa ---
def main():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        bot_token = config['TelegramBot']['bot_token']
        chat_id = config['TelegramBot']['chat_id']
        gnews_api_key = config['NewsAPI']['gnews_api_key']
    except (KeyError, FileNotFoundError):
        logging.critical("B£¥D: Upewnij siê, ¿e plik config.ini istnieje i jest kompletny.")
        return

    if "TWOJ_" in bot_token or "TWOJ_" in chat_id or "TWOJ_" in gnews_api_key:
        logging.critical("B£¥D: Uzupe³nij swoje klucze API w pliku config.ini!")
        return

    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("potwierdzam", confirm_action))
    application.add_handler(CommandHandler("ignoruje", ignore_action))
    application.add_handler(CommandHandler("reset", reset_state))
    
    job_data = {'news_api_key': gnews_api_key}
    application.job_queue.run_repeating(analysis_job, interval=CHECK_INTERVAL_SECONDS, first=10, chat_id=int(chat_id), data=job_data)
    
    logging.info("Bot Telegram zosta³ uruchomiony.")
    application.run_polling()

if __name__ == "__main__":
    main()