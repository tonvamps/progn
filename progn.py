import requests
import pandas as pd
import numpy as np
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ConversationHandler,
)
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Замените 'YOUR_TELEGRAM_BOT_TOKEN' на токен вашего бота
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

# Определение состояний для ConversationHandler
(
    SELECT_SIGNAL_COIN,
    SELECT_SIGNAL_TYPE,
    SET_PRICE_CHANGE_PARAMS,
    SET_TIME_FRAME,
    CONFIRM_SIGNAL,
) = range(5)

# Функция для получения индекса страха и жадности
def get_fear_and_greed_index():
    url = 'https://api.alternative.me/fng/'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            index_value = int(data['data'][0]['value'])
            return index_value
        else:
            logging.error(f"Fear and Greed Index not in data: {data}")
            return None
    except Exception as e:
        logging.error(f"Error fetching Fear and Greed Index: {e}")
        return None

# Функция для получения топ-100 криптовалют
def get_top_coins():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        coin_dict = {}
        for coin in data:
            coin_dict[coin['symbol'].upper()] = coin['id']
        return coin_dict
    except Exception as e:
        logging.error(f"Error fetching top coins: {e}")
        return {}

# Функция для получения данных о цене и объёме
def get_price_data(coin, days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart'
    max_days_allowed = 365
    params_days = min(days, max_days_allowed)
    params = {
        'vs_currency': 'usd',
        'days': params_days,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'prices' not in data or 'total_volumes' not in data:
            logging.error(f"Prices or volumes not in data: {data}")
            return None
        prices = data['prices']
        volumes = data['total_volumes']
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Добавляем столбцы high и low для некоторых индикаторов
        df['high'] = df['price']
        df['low'] = df['price']
        return df
    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return None

# Функция для получения настроения новостей (заглушка)
def get_news_sentiment(coin_name):
    # Заглушка функции. В реальной реализации используйте API новостей.
    sentiment_summary = "Нейтральный"
    sentiment_score = 0
    return sentiment_score, sentiment_summary

# Функция для получения клавиатуры главного меню
def get_main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔮 Рассчитать прогноз", callback_data='calculate')],
        [InlineKeyboardButton("🪙 Выбрать монету", callback_data='select_coin')],
        [InlineKeyboardButton("📆 Выбрать период", callback_data='select_period')],
        [InlineKeyboardButton("🔔 Настроить сигналы", callback_data='configure_signals')],
        [InlineKeyboardButton("📋 Мои сигналы", callback_data='view_signals')],
    ]
    return InlineKeyboardMarkup(keyboard)

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    context.user_data['selected_coin'] = 'bitcoin'
    context.user_data['selected_period'] = '1_day'
    reply_markup = get_main_menu_keyboard()
    await update.message.reply_text(
        '👋 Добро пожаловать! Выберите действие:', reply_markup=reply_markup
    )

# Обработчик нажатий кнопок
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    data = query.data

    # Проверяем, загружен ли список монет
    if 'coin_dict' not in context.bot_data:
        context.bot_data['coin_dict'] = get_top_coins()
    coin_dict = context.bot_data['coin_dict']

    if data == 'calculate':
        coin = context.user_data.get('selected_coin', 'bitcoin')
        period = context.user_data.get('selected_period', '1_day')
        days_map = {
            '1_day': 1,
            '3_days': 3,
            '5_days': 5,
            '7_days': 7,
            '30_days': 30,
            '365_days': 365,
        }
        forecast_days = days_map.get(period, 1)
        # Получаем достаточное количество исторических данных для анализа
        historical_days_needed = 365
        df = get_price_data(coin, days=historical_days_needed)
        if df is None or df.empty:
            await query.edit_message_text(
                text="Ошибка при получении данных о цене. Пожалуйста, попробуйте позже.",
                reply_markup=get_main_menu_keyboard(),
            )
            return
        coin_name = coin_dict.get(coin.upper(), coin)
        prediction = analyze_data(df, coin_name, forecast_days=forecast_days)
        # Получаем тикер монеты
        coin_ticker = [k for k, v in coin_dict.items() if v == coin]
        if coin_ticker:
            coin_ticker = coin_ticker[0]
        else:
            coin_ticker = coin.capitalize()
        await query.edit_message_text(
            text=(
                f"📊 Прогноз для {coin_ticker.upper()} на период {forecast_days} дней:\n"
                f"{prediction}"
            ),
            reply_markup=get_main_menu_keyboard(),
        )

    elif data == 'select_coin':
        letters = sorted(set(symbol[0].upper() for symbol in coin_dict.keys()))
        keyboard = [
            [InlineKeyboardButton(letter, callback_data=f'select_letter_{letter}')]
            for letter in letters
        ]
        keyboard.append([InlineKeyboardButton('🔙 Назад', callback_data='back_to_main')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text='🪙 Выберите первую букву монеты:', reply_markup=reply_markup
        )

    elif data.startswith('select_letter_'):
        letter = data[len('select_letter_'):]
        coins = [symbol for symbol in sorted(coin_dict.keys()) if symbol.startswith(letter)]
        if not coins:
            await query.edit_message_text(
                text='Монет на эту букву не найдено.', reply_markup=get_main_menu_keyboard()
            )
        else:
            keyboard = [
                [InlineKeyboardButton(f"{symbol}", callback_data=f'coin_{symbol}')]
                for symbol in coins
            ]
            keyboard.append([InlineKeyboardButton('🔙 Назад', callback_data='select_coin')])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                text=f'🪙 Выберите монету, начинающуюся с "{letter}":',
                reply_markup=reply_markup,
            )

    elif data.startswith('coin_'):
        selected_ticker = data[len('coin_'):]
        if selected_ticker in coin_dict:
            selected_coin = coin_dict[selected_ticker]
            context.user_data['selected_coin'] = selected_coin
            await query.edit_message_text(
                text=f"✅ Вы выбрали: {selected_ticker.upper()}",
                reply_markup=get_main_menu_keyboard(),
            )
        else:
            await query.edit_message_text(
                text="Монета не найдена.", reply_markup=get_main_menu_keyboard()
            )

    elif data == 'select_period':
        keyboard = [
            [InlineKeyboardButton('1 день', callback_data='period_1_day')],
            [InlineKeyboardButton('3 дня', callback_data='period_3_days')],
            [InlineKeyboardButton('5 дней', callback_data='period_5_days')],
            [InlineKeyboardButton('1 неделя', callback_data='period_7_days')],
            [InlineKeyboardButton('1 месяц', callback_data='period_30_days')],
            [InlineKeyboardButton('1 год', callback_data='period_365_days')],
            [InlineKeyboardButton('🔙 Назад', callback_data='back_to_main')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            text='📆 Выберите период прогнозирования:', reply_markup=reply_markup
        )

    elif data.startswith('period_'):
        period = data[len('period_'):]
        context.user_data['selected_period'] = period
        days_map = {
            '1_day': '1 день',
            '3_days': '3 дня',
            '5_days': '5 дней',
            '7_days': '1 неделя',
            '30_days': '1 месяц',
            '365_days': '1 год',
        }
        period_text = days_map.get(period, period)
        await query.edit_message_text(
            text=f"✅ Период прогнозирования установлен на: {period_text}",
            reply_markup=get_main_menu_keyboard(),
        )

    elif data == 'configure_signals':
        # Запускаем ConversationHandler для настройки сигналов
        await add_signal_start(update, context)

    elif data == 'view_signals':
        await view_signals(update, context)

    elif data == 'back_to_main':
        await query.edit_message_text(
            text="👋 Добро пожаловать! Выберите действие:", reply_markup=get_main_menu_keyboard()
        )

    else:
        await query.edit_message_text(
            text="Произошла ошибка. Пожалуйста, попробуйте снова.",
            reply_markup=get_main_menu_keyboard(),
        )

# Обработчик для начала настройки сигнала
async def add_signal_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    # Запускаем выбор монеты для сигнала
    await query.edit_message_text(
        text='🪙 Выберите монету для сигнала:',
        reply_markup=get_coin_selection_keyboard(context),
    )
    return SELECT_SIGNAL_COIN

def get_coin_selection_keyboard(context):
    coin_dict = context.bot_data['coin_dict']
    letters = sorted(set(symbol[0].upper() for symbol in coin_dict.keys()))
    keyboard = [
        [InlineKeyboardButton(letter, callback_data=f'select_signal_letter_{letter}')]
        for letter in letters
    ]
    keyboard.append([InlineKeyboardButton('🔙 Назад', callback_data='back_to_main')])
    return InlineKeyboardMarkup(keyboard)

async def select_signal_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    coin_dict = context.bot_data['coin_dict']
    data = query.data
    if data.startswith('select_signal_letter_'):
        letter = data[len('select_signal_letter_'):]
        coins = [symbol for symbol in sorted(coin_dict.keys()) if symbol.startswith(letter)]
        if not coins:
            await query.answer("Монет на эту букву не найдено.")
            return SELECT_SIGNAL_COIN
        keyboard = [
            [InlineKeyboardButton(f"{symbol}", callback_data=f'signal_coin_{symbol}')]
            for symbol in coins
        ]
        keyboard.append(
            [InlineKeyboardButton('🔙 Назад', callback_data='configure_signals_back')]
        )
        await query.edit_message_text(
            text=f'🪙 Выберите монету для сигнала, начинающуюся с "{letter}":',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return SELECT_SIGNAL_COIN
    elif data.startswith('signal_coin_'):
        selected_ticker = data[len('signal_coin_'):]
        if selected_ticker in coin_dict:
            selected_coin = coin_dict[selected_ticker]
            context.user_data['signal_setup'] = {'coin': selected_coin}
            await query.edit_message_text(
                text="Выберите тип сигнала:",
                reply_markup=get_signal_type_keyboard(),
            )
            return SELECT_SIGNAL_TYPE
        else:
            await query.answer("Монета не найдена.")
            return SELECT_SIGNAL_COIN
    elif data == 'configure_signals_back':
        await query.edit_message_text(
            text="🔔 Настройка сигналов:", reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    else:
        await query.answer("Неверный выбор.")
        return SELECT_SIGNAL_COIN

def get_signal_type_keyboard():
    keyboard = [
        [
            InlineKeyboardButton(
                'Изменение цены на X%', callback_data='signal_type_price_change'
            )
        ],
        [InlineKeyboardButton('🔙 Назад', callback_data='add_signal_back')],
    ]
    return InlineKeyboardMarkup(keyboard)

async def select_signal_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == 'signal_type_price_change':
        context.user_data['signal_setup']['type'] = 'price_change'
        await query.edit_message_text(
            text="Введите процент изменения цены для сигнала (например, 10 для 10%):"
        )
        return SET_PRICE_CHANGE_PARAMS
    elif data == 'add_signal_back':
        await add_signal_start(update, context)
        return SELECT_SIGNAL_COIN
    else:
        await query.answer("Неверный выбор.")
        return SELECT_SIGNAL_TYPE

async def set_price_change_params(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        percentage = float(update.message.text)
        context.user_data['signal_setup']['percentage'] = abs(percentage)
        await update.message.reply_text(
            "Выберите временной интервал для сигнала:",
            reply_markup=get_time_frame_keyboard(),
        )
        return SET_TIME_FRAME
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное число для процента изменения цены."
        )
        return SET_PRICE_CHANGE_PARAMS

def get_time_frame_keyboard():
    keyboard = [
        [InlineKeyboardButton('1 час', callback_data='time_frame_1h')],
        [InlineKeyboardButton('4 часа', callback_data='time_frame_4h')],
        [InlineKeyboardButton('12 часов', callback_data='time_frame_12h')],
        [InlineKeyboardButton('1 день', callback_data='time_frame_24h')],
    ]
    return InlineKeyboardMarkup(keyboard)

async def set_time_frame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data.startswith('time_frame_'):
        time_frame = data[len('time_frame_'):]
        context.user_data['signal_setup']['time_frame'] = time_frame
        await query.edit_message_text(
            text=get_signal_confirmation_text(context), reply_markup=get_confirmation_keyboard()
        )
        return CONFIRM_SIGNAL
    else:
        await query.answer("Неверный выбор.")
        return SET_TIME_FRAME

def get_signal_confirmation_text(context):
    coin_dict = context.bot_data['coin_dict']
    coin = context.user_data['signal_setup']['coin']
    coin_ticker = [k for k, v in coin_dict.items() if v == coin]
    if coin_ticker:
        coin_ticker = coin_ticker[0].upper()
    else:
        coin_ticker = coin.capitalize()
    percentage = context.user_data['signal_setup']['percentage']
    time_frame = context.user_data['signal_setup']['time_frame']
    time_frame_texts = {
        '1h': '1 час',
        '4h': '4 часа',
        '12h': '12 часов',
        '24h': '1 день',
    }
    time_frame_text = time_frame_texts.get(time_frame, time_frame)
    text = (
        f"⚙️ Параметры сигнала:\n"
        f"Монета: {coin_ticker}\n"
        f"Изменение цены: {percentage}%\n"
        f"Временной интервал: {time_frame_text}\n\n"
        f"Сохранить этот сигнал?"
    )
    return text

def get_confirmation_keyboard():
    keyboard = [
        [InlineKeyboardButton('✅ Да', callback_data='confirm_signal_yes')],
        [InlineKeyboardButton('❌ Нет', callback_data='confirm_signal_no')],
    ]
    return InlineKeyboardMarkup(keyboard)

async def confirm_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == 'confirm_signal_yes':
        if 'signals' not in context.user_data:
            context.user_data['signals'] = []
        context.user_data['signals'].append(context.user_data['signal_setup'])
        context.user_data.pop('signal_setup', None)
        await query.edit_message_text(
            text="✅ Сигнал сохранен.", reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    elif data == 'confirm_signal_no':
        context.user_data.pop('signal_setup', None)
        await query.edit_message_text(
            text="❌ Настройка сигнала отменена.", reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    else:
        await query.answer("Неверный выбор.")
        return CONFIRM_SIGNAL

async def view_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    coin_dict = context.bot_data['coin_dict']
    user_signals = context.user_data.get('signals', [])
    if not user_signals:
        await query.edit_message_text(
            text="📋 У вас нет настроенных сигналов.", reply_markup=get_main_menu_keyboard()
        )
    else:
        keyboard = []
        for idx, signal in enumerate(user_signals):
            coin = signal.get('coin', 'N/A')
            coin_ticker = [k for k, v in coin_dict.items() if v == coin]
            if coin_ticker:
                coin_ticker = coin_ticker[0].upper()
            else:
                coin_ticker = coin.capitalize()
            signal_type = signal.get('type', 'N/A')
            if signal_type == 'price_change':
                percentage = signal.get('percentage', 'N/A')
                time_frame = signal.get('time_frame', 'N/A')
                time_frame_texts = {
                    '1h': '1 час',
                    '4h': '4 часа',
                    '12h': '12 часов',
                    '24h': '1 день',
                }
                time_frame_text = time_frame_texts.get(time_frame, time_frame)
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            f"{coin_ticker}: Изм. на {percentage}% за {time_frame_text}",
                            callback_data=f'delete_signal_{idx}',
                        )
                    ]
                )
        keyboard.append([InlineKeyboardButton('🔙 Назад', callback_data='back_to_main')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="📋 Ваши сигналы:", reply_markup=reply_markup)

async def delete_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data.startswith('delete_signal_'):
        idx = int(data[len('delete_signal_') :])
        user_signals = context.user_data.get('signals', [])
        if 0 <= idx < len(user_signals):
            user_signals.pop(idx)
            await query.edit_message_text(
                text="🗑️ Сигнал удален.", reply_markup=get_main_menu_keyboard()
            )
        else:
            await query.answer("Ошибка: сигнал не найден.")
    else:
        await query.answer("Неверный выбор.")

# Функция для анализа волн Эллиота
def elliott_wave_analysis(df):
    # Определяем экстремумы
    order = 5  # Параметр чувствительности
    df['min'] = df.iloc[
        argrelextrema(df['price'].values, np.less_equal, order=order)[0]
    ]['price']
    df['max'] = df.iloc[
        argrelextrema(df['price'].values, np.greater_equal, order=order)[0]
    ]['price']
    # Создаем список экстремумов
    extrema = df[['min', 'max']].dropna(how='all')
    extrema['type'] = extrema.apply(
        lambda row: 'min' if not pd.isna(row['min']) else 'max', axis=1
    )
    extrema['price'] = extrema.apply(
        lambda row: row['min'] if not pd.isna(row['min']) else row['max'], axis=1
    )
    extrema = extrema[['price', 'type']]
    # Проверяем наличие достаточного количества точек
    if len(extrema) < 9:
        return "📉 Недостаточно данных для анализа волн Эллиота."
    # Ищем паттерны волн Эллиота
    last_patterns = []
    for i in range(len(extrema) - 8):
        pattern = extrema.iloc[i : i + 9]
        types = pattern['type'].tolist()
        # Проверяем последовательность типов экстремумов
        if types == [
            'min',
            'max',
            'min',
            'max',
            'min',
            'max',
            'min',
            'max',
            'min',
        ]:
            last_patterns.append(pattern)
    # Анализируем последний найденный паттерн
    if last_patterns:
        last_pattern = last_patterns[-1]
        prices = last_pattern['price'].values
        # Рассчитываем отношения Фибоначчи между волнами
        wave1 = prices[1] - prices[0]
        wave3 = prices[3] - prices[2]
        wave5 = prices[5] - prices[4]
        # Проверяем соотношения волн по правилам Эллиота
        if abs(wave3) > abs(wave1) and abs(wave5) < abs(wave3):
            # Анализируем объемы торгов
            recent_volumes = df['volume'].iloc[-len(last_pattern) :]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            if recent_volumes.mean() > avg_volume:
                return "Импульсная волна завершается, возможна коррекция (📉 медвежий сигнал)."
            else:
                return "Импульсная волна продолжается (📈 бычий сигнал)."
        else:
            return "Коррекционная волна, возможен разворот тренда."
    else:
        return "📉 Не удалось определить волны Эллиота."

# Функция для анализа данных и стратегий
def analyze_data(df, coin_name, forecast_days):
    if df is None or df.empty:
        return "Нет достаточных данных для анализа."
    # Проверяем, достаточно ли данных
    if len(df) < 100:
        return "Недостаточно данных для анализа."
    # Вычисляем технические индикаторы
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    df['EMA_20'] = EMAIndicator(close=df['price'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['price'], window=50).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['price'], window=14).rsi()
    macd = MACD(close=df['price'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    # Полосы Боллинджера
    bb = BollingerBands(close=df['price'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    # Индекс товарного канала (CCI)
    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['price'], window=20)
    df['CCI'] = cci.cci()
    # Стохастический осциллятор
    stoch = StochasticOscillator(
        high=df['high'], low=df['low'], close=df['price'], window=14
    )
    df['STOCHk'] = stoch.stoch()
    df['STOCHd'] = stoch.stoch_signal()
    # Средний истинный диапазон (ATR)
    atr = AverageTrueRange(
        high=df['high'], low=df['low'], close=df['price'], window=14
    )
    df['ATR'] = atr.average_true_range()
    # Индикатор объема баланса (OBV)
    obv_indicator = OnBalanceVolumeIndicator(
        close=df['price'], volume=df['volume']
    )
    df['OBV'] = obv_indicator.on_balance_volume()
    # Средний индекс направленного движения (ADX)
    adx_indicator = ADXIndicator(
        high=df['high'], low=df['low'], close=df['price'], window=14
    )
    df['ADX'] = adx_indicator.adx()
    # Генерируем сигналы на основе индикаторов
    signals = []
    bullish_weighted_signals = 0
    bearish_weighted_signals = 0
    # Пересечение скользящих средних
    weight_ma = 1
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
        signals.append("📈 Короткосрочный SMA выше долгосрочного SMA (бычий сигнал).")
        bullish_weighted_signals += weight_ma
    else:
        signals.append("📉 Короткосрочный SMA ниже долгосрочного SMA (медвежий сигнал).")
        bearish_weighted_signals += weight_ma
    if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
        signals.append("📈 Короткосрочный EMA выше долгосрочного EMA (бычий сигнал).")
        bullish_weighted_signals += weight_ma
    else:
        signals.append("📉 Короткосрочный EMA ниже долгосрочного EMA (медвежий сигнал).")
        bearish_weighted_signals += weight_ma
    # Сигнал RSI
    weight_rsi = 1
    if df['RSI'].iloc[-1] < 30:
        signals.append("📈 RSI указывает на перепроданность (бычий сигнал).")
        bullish_weighted_signals += weight_rsi
    elif df['RSI'].iloc[-1] > 70:
        signals.append("📉 RSI указывает на перекупленность (медвежий сигнал).")
        bearish_weighted_signals += weight_rsi
    else:
        signals.append("⚪️ RSI находится в нормальном диапазоне.")
    # Сигнал MACD
    weight_macd = 1
    if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        signals.append("📈 MACD выше сигнальной линии (бычий сигнал).")
        bullish_weighted_signals += weight_macd
    else:
        signals.append("📉 MACD ниже сигнальной линии (медвежий сигнал).")
        bearish_weighted_signals += weight_macd
    # Сигнал Bollinger Bands
    weight_bb = 1
    if df['price'].iloc[-1] < df['BB_lower'].iloc[-1]:
        signals.append("📈 Цена ниже нижней полосы Bollinger Bands (бычий сигнал).")
        bullish_weighted_signals += weight_bb
    elif df['price'].iloc[-1] > df['BB_upper'].iloc[-1]:
        signals.append("📉 Цена выше верхней полосы Bollinger Bands (медвежий сигнал).")
        bearish_weighted_signals += weight_bb
    else:
        signals.append("⚪️ Цена внутри полос Bollinger Bands.")
    # Сигнал CCI
    weight_cci = 1
    if df['CCI'].iloc[-1] < -100:
        signals.append("📈 CCI указывает на перепроданность (бычий сигнал).")
        bullish_weighted_signals += weight_cci
    elif df['CCI'].iloc[-1] > 100:
        signals.append("📉 CCI указывает на перекупленность (медвежий сигнал).")
        bearish_weighted_signals += weight_cci
    else:
        signals.append("⚪️ CCI находится в нормальном диапазоне.")
    # Сигнал Стохастик
    weight_stoch = 1
    if df['STOCHk'].iloc[-1] < 20:
        signals.append("📈 Стохастик указывает на перепроданность (бычий сигнал).")
        bullish_weighted_signals += weight_stoch
    elif df['STOCHk'].iloc[-1] > 80:
        signals.append("📉 Стохастик указывает на перекупленность (медвежий сигнал).")
        bearish_weighted_signals += weight_stoch
    else:
        signals.append("⚪️ Стохастик в нормальном диапазоне.")
    # Сигнал ADX
    weight_adx = 1
    if df['ADX'].iloc[-1] > 25:
        signals.append("📈 ADX указывает на сильный тренд.")
        bullish_weighted_signals += weight_adx
    else:
        signals.append("📉 ADX указывает на слабый тренд.")
        bearish_weighted_signals += weight_adx
    # Анализ волн Эллиота
    weight_elliott = 2
    elliott_wave_result = elliott_wave_analysis(df)
    signals.append(f"🌊 Анализ волн Эллиота: {elliott_wave_result}")
    if "бычий сигнал" in elliott_wave_result:
        bullish_weighted_signals += weight_elliott
    elif "медвежий сигнал" in elliott_wave_result:
        bearish_weighted_signals += weight_elliott
    # Анализ объема торгов
    weight_volume = 1
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    if df['volume'].iloc[-1] > avg_volume:
        signals.append("📈 Объем торгов выше среднего (подтверждение тренда).")
        bullish_weighted_signals += weight_volume
    else:
        signals.append("📉 Объем торгов ниже среднего (возможная слабость тренда).")
        bearish_weighted_signals += weight_volume
    # Анализ новостей
    weight_news = 2
    sentiment_score, sentiment_summary = get_news_sentiment(coin_name)
    signals.append(f"📰 Новостное настроение: {sentiment_summary}")
    if sentiment_score > 0:
        bullish_weighted_signals += weight_news
    elif sentiment_score < 0:
        bearish_weighted_signals += weight_news
    # Индекс страха и жадности
    weight_fgi = 1
    fear_greed_index = get_fear_and_greed_index()
    if fear_greed_index is not None:
        signals.append(f"📊 Индекс страха и жадности: {fear_greed_index}")
        if fear_greed_index < 40:
            signals.append("📈 Рынок в страхе (возможность покупки).")
            bullish_weighted_signals += weight_fgi
        elif fear_greed_index > 60:
            signals.append("📉 Рынок в жадности (возможность продажи).")
            bearish_weighted_signals += weight_fgi
    else:
        signals.append("⚠️ Не удалось получить индекс страха и жадности.")
    # Вычисляем вероятности
    total_weighted_signals = bullish_weighted_signals + bearish_weighted_signals
    if total_weighted_signals == 0:
        bullish_probability = 50
        bearish_probability = 50
    else:
        bullish_probability = (bullish_weighted_signals / total_weighted_signals) * 100
        bearish_probability = (bearish_weighted_signals / total_weighted_signals) * 100
    # Рассчитываем прогнозируемую цену
    last_price = df['price'].iloc[-1]
    df['daily_return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    # Рассчитываем среднюю логарифмическую доходность и стандартное отклонение
    avg_log_return = df['log_return'].mean()
    std_log_return = df['log_return'].std()
    # Определяем фактор настроения на основе сигналов
    if total_weighted_signals == 0:
        sentiment_factor = 0
    else:
        sentiment_factor = (
            bullish_weighted_signals - bearish_weighted_signals
        ) / total_weighted_signals
    # Корректируем среднюю доходность
    adjusted_log_return = avg_log_return + sentiment_factor * std_log_return
    # Прогнозируемая цена
    forecasted_price = last_price * np.exp(adjusted_log_return * forecast_days)
    # Ожидаемое процентное изменение цены
    expected_percentage_change = ((forecasted_price - last_price) / last_price) * 100
    # Проверяем корректность прогнозируемой цены
    if bearish_probability > bullish_probability and forecasted_price > last_price:
        forecasted_price = last_price * np.exp(-abs(adjusted_log_return) * forecast_days)
        expected_percentage_change = ((forecasted_price - last_price) / last_price) * 100
    elif bullish_probability > bearish_probability and forecasted_price < last_price:
        forecasted_price = last_price * np.exp(abs(adjusted_log_return) * forecast_days)
        expected_percentage_change = ((forecasted_price - last_price) / last_price) * 100
    # Округляем прогнозируемую цену и процентное изменение
    forecasted_price = round(forecasted_price, 4)
    expected_percentage_change = round(expected_percentage_change, 2)
    prediction = '\n'.join(signals)
    conclusion = (
        f"\n\n📈 Вероятность роста цены: {bullish_probability:.1f}%"
        f"\n📉 Вероятность падения цены: {bearish_probability:.1f}%"
    )
    price_info = f"\n\n💰 Текущая цена {coin_name.upper()}: ${last_price:.4f}"
    forecast_info = (
        f"\n💱 Прогнозируемая цена через {forecast_days} дней: "
        f"${forecasted_price} ({expected_percentage_change:+.2f}%)"
    )
    return prediction + conclusion + price_info + forecast_info

# Функция для проверки пользовательских сигналов и отправки уведомлений
async def check_user_signals(context: ContextTypes.DEFAULT_TYPE):
    coin_dict = context.bot_data['coin_dict']
    for user_id, user_data in context.application.user_data.items():
        if 'signals' in user_data:
            for signal in user_data['signals']:
                coin = signal['coin']
                percentage = signal['percentage']
                time_frame = signal['time_frame']
                df = get_price_data(coin, days=1)
                if df is None or df.empty:
                    continue
                current_price = df['price'].iloc[-1]
                time_deltas = {
                    '1h': pd.Timedelta(hours=1),
                    '4h': pd.Timedelta(hours=4),
                    '12h': pd.Timedelta(hours=12),
                    '24h': pd.Timedelta(hours=24),
                }
                time_delta = time_deltas.get(time_frame, pd.Timedelta(hours=1))
                past_time = df.index[-1] - time_delta
                past_prices = df[df.index <= past_time]
                if past_prices.empty:
                    continue
                past_price = past_prices['price'].iloc[-1]
                price_change = (current_price - past_price) / past_price * 100
                if abs(price_change) >= percentage:
                    coin_ticker = [k for k, v in coin_dict.items() if v == coin]
                    if coin_ticker:
                        coin_ticker = coin_ticker[0].upper()
                    else:
                        coin_ticker = coin.capitalize()
                    direction = 'выросла' if price_change > 0 else 'упала'
                    time_frame_texts = {
                        '1h': '1 час',
                        '4h': '4 часа',
                        '12h': '12 часов',
                        '24h': '1 день',
                    }
                    time_frame_text = time_frame_texts.get(time_frame, time_frame)
                    message = (
                        f"🚨 Цена {coin_ticker} {direction} на {price_change:.2f}% "
                        f"за последние {time_frame_text}!"
                    )
                    await context.bot.send_message(chat_id=user_id, text=message)
                    # Опционально, можно удалить сигнал после срабатывания
                    # user_data['signals'].remove(signal)

# Основная функция
def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    # Сохраняем список монет в bot_data
    application.bot_data['coin_dict'] = get_top_coins()

    # Обработчики команд
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', start))

    # Обработчик кнопок
    application.add_handler(CallbackQueryHandler(button))

    # ConversationHandler для настройки сигналов
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(add_signal_start, pattern='^configure_signals$')],
        states={
            SELECT_SIGNAL_COIN: [
                CallbackQueryHandler(select_signal_coin, pattern='^select_signal_letter_'),
                CallbackQueryHandler(select_signal_coin, pattern='^signal_coin_'),
                CallbackQueryHandler(select_signal_coin, pattern='^configure_signals_back$'),
            ],
            SELECT_SIGNAL_TYPE: [
                CallbackQueryHandler(select_signal_type, pattern='^signal_type_price_change$'),
                CallbackQueryHandler(select_signal_type, pattern='^add_signal_back$'),
            ],
            SET_PRICE_CHANGE_PARAMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_price_change_params)],
            SET_TIME_FRAME: [CallbackQueryHandler(set_time_frame, pattern='^time_frame_')],
            CONFIRM_SIGNAL: [
                CallbackQueryHandler(confirm_signal, pattern='^confirm_signal_yes$'),
                CallbackQueryHandler(confirm_signal, pattern='^confirm_signal_no$'),
            ],
        },
        fallbacks=[CommandHandler('cancel', start)],
    )
    application.add_handler(conv_handler)

    # Обработчик для удаления сигналов
    application.add_handler(CallbackQueryHandler(delete_signal, pattern='^delete_signal_'))

    # Обработчик текстовых сообщений (для ввода процентов)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, set_price_change_params))

    # Проверяем пользовательские сигналы каждые 5 минут
    application.job_queue.run_repeating(check_user_signals, interval=300, first=0)

    application.run_polling()

if __name__ == '__main__':
    main()
