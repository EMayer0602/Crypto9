import os
import math
import json
import shutil
from pathlib import Path
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from ta.volatility import AverageTrueRange
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def _load_env_file(path: str = ".env") -> None:
	env_path = Path(path)
	if not env_path.is_absolute():
		env_path = Path(__file__).resolve().parent / env_path
	if not env_path.exists():
		return
	try:
		for raw_line in env_path.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip()
			if key and value:
				os.environ[key] = value
	except OSError as exc:
		print(f"[Env] Failed to load {env_path}: {exc}")


_load_env_file()

BERLIN_TZ = ZoneInfo("Europe/Berlin")



def _truthy(value) -> bool:
	return str(value).strip().lower() in {"1", "true", "yes", "on"}


USE_TESTNET = _truthy(os.getenv("BINANCE_USE_TESTNET", "false"))

def timeframe_to_minutes(tf_str: str) -> int:
	unit = tf_str[-1].lower()
	value = int(tf_str[:-1])
	if unit == "m":
		return value
	if unit == "h":
		return value * 60
	if unit == "d":
		return value * 1440
	raise ValueError(f"Unsupported timeframe unit in {tf_str}")


EXCHANGE_ID = "binance"
TIMEFRAME = "1h"
LOOKBACK = 8760  # 365 days × 24h for full year data
SYMBOLS = [
	"BTC/EUR",
	"ETH/EUR",
	"XRP/EUR",
	# "ADA/EUR",
	# "BNB/EUR",
	"LINK/EUR",
	# "DOGE/EUR",
	"LUNC/USDT",
	"SOL/EUR",
	"SUI/EUR",
	"TNSR/USDC",
	# "ZEC/EUR",
	"ZEC/USDC",
]

RUN_PARAMETER_SWEEP = False
RUN_SAVED_PARAMS = False
RUN_OVERALL_BEST = True
ENABLE_LONGS = True
ENABLE_SHORTS = True  # Restored: 705 short trades in simulation

USE_MIN_HOLD_FILTER = True
DEFAULT_MIN_HOLD_DAYS = 0
MIN_HOLD_DAY_VALUES = [0, 1, 2]

USE_HIGHER_TIMEFRAME_FILTER = True
HIGHER_TIMEFRAME = "12h"
HTF_LOOKBACK = 1000  # Enough for full year at 12h timeframe
HTF_LENGTH = 20
HTF_FACTOR = 3.0
HTF_PSAR_STEP = 0.02
HTF_PSAR_MAX_STEP = 0.2
HTF_JMA_LENGTH = 30
HTF_JMA_PHASE = 0
HTF_KAMA_LENGTH = 20
HTF_KAMA_SLOW_LENGTH = 40
HTF_MAMA_FAST_LIMIT = 0.5
HTF_MAMA_SLOW_LIMIT = 0.05

USE_MOMENTUM_FILTER = False
MOMENTUM_TYPE = "RSI"
MOMENTUM_WINDOW = 14
RSI_LONG_THRESHOLD = 55
RSI_SHORT_THRESHOLD = 45

USE_BREAKOUT_FILTER = False
BREAKOUT_ATR_MULT = 1.5
BREAKOUT_REQUIRE_DIRECTION = True

START_EQUITY = 14000.0
RISK_FRACTION = 1
STAKE_DIVISOR = 14
FEE_RATE = 0.001
ATR_WINDOW = 14
ATR_STOP_MULTS = [None, 1.0, 1.5, 2.0]

BASE_OUT_DIR = "report_html"
BARS_PER_DAY = max(1, int(1440 / timeframe_to_minutes(TIMEFRAME)))
CLEAR_BASE_OUTPUT_ON_SWEEP = True

OVERALL_SUMMARY_HTML = os.path.join(BASE_OUT_DIR, "overall_best_results.html")
OVERALL_PARAMS_CSV = os.path.join(BASE_OUT_DIR, "best_params_overall.csv")
OVERALL_DETAILED_HTML = os.path.join(BASE_OUT_DIR, "overall_best_detailed.html")
OVERALL_FLAT_CSV = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.csv")
OVERALL_FLAT_JSON = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.json")
GLOBAL_BEST_RESULTS = {}

INDICATOR_PRESETS = {
	"supertrend": {
		"display_name": "Supertrend",
		"slug": "supertrend",
		"param_a_label": "Length",
		"param_b_label": "Factor",
		"param_a_values": [7, 10, 14],
		"param_b_values": [2.0, 3.0, 4.0],
		"default_a": 10,
		"default_b": 3.0,
	},
	"psar": {
		"display_name": "Parabolic SAR",
		"slug": "psar",
		"param_a_label": "Step",
		"param_b_label": "MaxStep",
		"param_a_values": [0.01, 0.02, 0.03],
		"param_b_values": [0.1, 0.2, 0.3],
		"default_a": 0.02,
		"default_b": 0.2,
	},
	"jma": {
		"display_name": "Jurik Moving Average",
		"slug": "jma",
		"param_a_label": "Length",
		"param_b_label": "Phase",
		"param_a_values": [20, 30, 50],
		"param_b_values": [-50, 0, 50],
		"default_a": 30,
		"default_b": 0,
	},
	"kama": {
		"display_name": "Kaufman AMA",
		"slug": "kama",
		"param_a_label": "Length",
		"param_b_label": "SlowLength",
		"param_a_values": [10, 20, 30],
		"param_b_values": [30, 40, 50],
		"default_a": 20,
		"default_b": 40,
	},
	"mama": {
		"display_name": "Mesa Adaptive MA",
		"slug": "mama",
		"param_a_label": "FastLimit",
		"param_b_label": "SlowLimit",
		"param_a_values": [0.5, 0.4, 0.3],
		"param_b_values": [0.05, 0.03, 0.01],
		"default_a": 0.5,
		"default_b": 0.05,
	},
}

ACTIVE_INDICATORS = ["jma", "kama", "supertrend"]

INDICATOR_TYPE = ""
INDICATOR_DISPLAY_NAME = ""
INDICATOR_SLUG = ""
PARAM_A_LABEL = ""
PARAM_B_LABEL = ""
PARAM_A_VALUES: list = []
PARAM_B_VALUES: list = []
DEFAULT_PARAM_A = 0
DEFAULT_PARAM_B = 0

OUT_DIR = BASE_OUT_DIR
REPORT_FILE = "supertrend_report.html"
BEST_PARAMS_FILE = "best_params.csv"

_exchange = None
_data_exchange = None
DATA_CACHE = {}
DATA_CACHE_TIMESTAMPS = {}
CACHE_TTL_SECONDS = 60  # 1 minute TTL - ensures fresh data each run


INDICATOR_CACHE = {}
INDICATOR_CACHE_MAX_SIZE = 100  # Maximum number of indicator results to cache


def clear_data_cache() -> None:
	"""Clear all cached data to force fresh fetches."""
	global DATA_CACHE, DATA_CACHE_TIMESTAMPS, INDICATOR_CACHE
	DATA_CACHE.clear()
	DATA_CACHE_TIMESTAMPS.clear()
	INDICATOR_CACHE.clear()


def _get_indicator_cache_key(indicator_type: str, df_hash: int, param_a: float, param_b: float) -> tuple:
	"""Generate a cache key for indicator results."""
	return (indicator_type, df_hash, param_a, param_b)


def _hash_dataframe(df: pd.DataFrame) -> int:
	"""Create a hash of a DataFrame for caching purposes."""
	if df.empty:
		return 0
	# Use the index and close column to create a fast hash
	try:
		return hash((df.index[0], df.index[-1], len(df), df["close"].iloc[-1]))
	except Exception:
		return 0


def _is_cache_expired(key: tuple) -> bool:
	"""Check if cached data has expired based on TTL."""
	if key not in DATA_CACHE_TIMESTAMPS:
		return True
	cached_time = DATA_CACHE_TIMESTAMPS[key]
	elapsed = (datetime.now(timezone.utc) - cached_time).total_seconds()
	return elapsed > CACHE_TTL_SECONDS


def configure_exchange(use_testnet=None) -> None:
	global USE_TESTNET, _exchange, _data_exchange
	if use_testnet is None or use_testnet == USE_TESTNET:
		return
	USE_TESTNET = use_testnet
	os.environ["BINANCE_USE_TESTNET"] = "1" if use_testnet else "0"
	_exchange = None
	_data_exchange = None


def _build_exchange(include_keys: bool):
	cls = getattr(ccxt, EXCHANGE_ID)
	args = {"enableRateLimit": True}
	if include_keys:
		api_key, api_secret = _current_api_credentials()
		if api_key and api_secret:
			args.update({"apiKey": api_key, "secret": api_secret})
	exchange = cls(args)
	options = dict(getattr(exchange, "options", {}))
	options["warnOnFetchCurrenciesWithoutPermission"] = False
	exchange.options = options
	if hasattr(exchange, "has") and isinstance(exchange.has, dict):
		exchange.has["fetchCurrencies"] = False
	if USE_TESTNET and hasattr(exchange, "set_sandbox_mode"):
		try:
			exchange.set_sandbox_mode(True)
		except Exception as exc:
			print(f"[Exchange] Failed to enable Binance sandbox mode: {exc}")
	return exchange


def _current_api_credentials():
	if USE_TESTNET:
		return (
			os.getenv("BINANCE_API_KEY_TEST"),
			os.getenv("BINANCE_API_SECRET_TEST"),
		)
	return (
		os.getenv("BINANCE_API_KEY"),
		os.getenv("BINANCE_API_SECRET"),
	)


def clear_directory(path: str) -> None:
	if not os.path.isdir(path):
		return
	for entry in os.listdir(path):
		full_path = os.path.join(path, entry)
		if os.path.isdir(full_path):
			shutil.rmtree(full_path, ignore_errors=True)
		else:
			try:
				os.remove(full_path)
			except OSError:
				pass


def clear_sweep_targets(indicator_names, htf_values):
	if not indicator_names or not htf_values:
		return
	for indicator_name in indicator_names:
		preset = INDICATOR_PRESETS.get(indicator_name)
		if not preset:
			continue
		slug = preset.get("slug", indicator_name)
		for htf_value in htf_values:
			htf_clean = str(htf_value).replace("/", "")
			folder = os.path.join(BASE_OUT_DIR, f"{slug}_{htf_clean}")
			clear_directory(folder)


def get_exchange():
	global _exchange
	if _exchange is None:
		_exchange = _build_exchange(include_keys=True)
	return _exchange


def get_data_exchange():
	global _data_exchange
	if _data_exchange is None:
		_data_exchange = _build_exchange(include_keys=False)
	return _data_exchange


def _fetch_direct_ohlcv(symbol, timeframe, limit):
	exchange = get_data_exchange()
	buffer = max(50, limit // 5)
	fetch_limit = limit + buffer
	ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
	cols = ["timestamp", "open", "high", "low", "close", "volume"]
	df = pd.DataFrame(ohlcv, columns=cols)
	df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
	return df.set_index("timestamp").tail(limit)


def _maybe_append_synthetic_bar(df, symbol, timeframe):
	try:
		tf_minutes = timeframe_to_minutes(timeframe)
	except ValueError:
		return df
	if tf_minutes <= 1:
		return df
	if df is None:
		df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
	now = pd.Timestamp.now(BERLIN_TZ)
	bucket = pd.Timedelta(minutes=tf_minutes)
	current_end = now.floor(f"{tf_minutes}min") + bucket
	if not df.empty:
		last_idx = df.index.max()
		if last_idx > current_end:
			return df
		if last_idx == current_end:
			df = df.drop(index=current_end)
	current_start = current_end - bucket
	minutes_needed = max(2, int(np.ceil((now - current_start).total_seconds() / 60.0)) + 2)
	try:
		minute_df = _fetch_direct_ohlcv(symbol, "1m", limit=minutes_needed)
	except Exception as exc:
		print(f"[Warn] Failed to fetch 1m data for {symbol}: {exc}")
		return df
	slice_df = minute_df[(minute_df.index > current_start) & (minute_df.index <= now)]
	if slice_df.empty:
		return df
	synthetic = pd.DataFrame({
		"open": float(slice_df["open"].iloc[0]),
		"high": float(slice_df["high"].max()),
		"low": float(slice_df["low"].min()),
		"close": float(slice_df["close"].iloc[-1]),
		"volume": float(slice_df["volume"].sum()),
	}, index=[current_end])
	combined = pd.concat([df, synthetic])
	combined = combined[~combined.index.duplicated(keep="last")]
	combined = combined.sort_index()
	return combined


def fetch_data(symbol, timeframe, limit):
	key = (symbol, timeframe, limit)
	# Check if cache is valid (exists and not expired)
	cache_valid = key in DATA_CACHE and not _is_cache_expired(key)

	if cache_valid:
		base_df = DATA_CACHE[key]
	else:
		cache_df = None
		exchange = get_data_exchange()
		supported_timeframes = getattr(exchange, "timeframes", {}) or {}
		if timeframe in supported_timeframes:
			cache_df = _fetch_direct_ohlcv(symbol, timeframe, limit)
		else:
			target_minutes = timeframe_to_minutes(timeframe)
			base_minutes = timeframe_to_minutes(TIMEFRAME)
			if target_minutes < base_minutes or target_minutes % base_minutes != 0:
				raise ValueError(f"Cannot synthesize timeframe {timeframe} from base {TIMEFRAME}")
			factor = target_minutes // base_minutes
			base_limit = limit * factor + 10
			base_df_source = fetch_data(symbol, TIMEFRAME, base_limit)
			if base_df_source.empty:
				cache_df = base_df_source
			else:
				agg_rule = f"{target_minutes}min"
				synth = base_df_source.resample(agg_rule, label="right", closed="right").agg({
					"open": "first",
					"high": "max",
					"low": "min",
					"close": "last",
					"volume": "sum",
				})
				synth = synth.dropna(subset=["open", "high", "low", "close"])
				cache_df = synth.tail(limit)
		DATA_CACHE[key] = cache_df
		DATA_CACHE_TIMESTAMPS[key] = datetime.now(timezone.utc)
		base_df = cache_df
	df_copy = base_df.copy() if base_df is not None else pd.DataFrame()
	df_with_live = _maybe_append_synthetic_bar(df_copy, symbol, timeframe)
	return df_with_live


def update_output_targets():
	global OUT_DIR, REPORT_FILE, BEST_PARAMS_FILE
	slug = INDICATOR_SLUG or "supertrend"
	htf = HIGHER_TIMEFRAME.replace("/", "")
	folder = f"{slug}_{htf}"
	OUT_DIR = os.path.join(BASE_OUT_DIR, folder)
	REPORT_FILE = f"long_strategy_report_{folder}.html"
	BEST_PARAMS_FILE = f"best_params_{folder}.csv"


def apply_indicator_type(name: str):
	global INDICATOR_TYPE, INDICATOR_DISPLAY_NAME, INDICATOR_SLUG
	global PARAM_A_LABEL, PARAM_B_LABEL, PARAM_A_VALUES, PARAM_B_VALUES
	global DEFAULT_PARAM_A, DEFAULT_PARAM_B
	preset = INDICATOR_PRESETS[name]
	INDICATOR_TYPE = name
	INDICATOR_DISPLAY_NAME = preset["display_name"]
	INDICATOR_SLUG = preset["slug"]
	PARAM_A_LABEL = preset["param_a_label"]
	PARAM_B_LABEL = preset["param_b_label"]
	PARAM_A_VALUES = preset["param_a_values"]
	PARAM_B_VALUES = preset["param_b_values"]
	DEFAULT_PARAM_A = preset["default_a"]
	DEFAULT_PARAM_B = preset["default_b"]
	update_output_targets()


def apply_higher_timeframe(htf_value: str):
	global HIGHER_TIMEFRAME
	HIGHER_TIMEFRAME = htf_value
	update_output_targets()


def get_indicator_candidates():
	if ACTIVE_INDICATORS:
		return [name for name in ACTIVE_INDICATORS if name in INDICATOR_PRESETS]
	return list(INDICATOR_PRESETS.keys())


def get_enabled_directions():
	directions = []
	if ENABLE_LONGS:
		directions.append("long")
	if ENABLE_SHORTS:
		directions.append("short")
	if not directions:
		directions.append("long")
	return directions


def get_highertimeframe_candidates():
	return [f"{hours}h" for hours in range(3, 25)]


def _compute_supertrend_numba(close_arr, high_arr, low_arr, basic_ub_arr, basic_lb_arr):
	"""Optimized Supertrend calculation using NumPy arrays.

	This vectorized implementation is ~10x faster than row-by-row iteration.
	"""
	n = len(close_arr)
	final_ub = np.empty(n)
	final_lb = np.empty(n)
	supertrend = np.empty(n)
	trend = np.empty(n, dtype=np.int32)

	# Initialize first values
	final_ub[0] = basic_ub_arr[0]
	final_lb[0] = basic_lb_arr[0]
	trend[0] = 1 if close_arr[0] >= final_lb[0] else -1
	supertrend[0] = final_lb[0] if trend[0] == 1 else final_ub[0]

	# Optimized loop with NumPy arrays (avoids pandas indexing overhead)
	for i in range(1, n):
		prev_close = close_arr[i - 1]

		# Calculate final upper/lower bands
		if basic_ub_arr[i] < final_ub[i - 1] or prev_close > final_ub[i - 1]:
			final_ub[i] = basic_ub_arr[i]
		else:
			final_ub[i] = final_ub[i - 1]

		if basic_lb_arr[i] > final_lb[i - 1] or prev_close < final_lb[i - 1]:
			final_lb[i] = basic_lb_arr[i]
		else:
			final_lb[i] = final_lb[i - 1]

		# Determine trend direction
		close = close_arr[i]
		prev_trend = trend[i - 1]

		if prev_trend == 1:
			if close <= final_lb[i]:
				trend[i] = -1
				supertrend[i] = final_ub[i]
			else:
				trend[i] = 1
				supertrend[i] = final_lb[i]
		else:
			if close >= final_ub[i]:
				trend[i] = 1
				supertrend[i] = final_lb[i]
			else:
				trend[i] = -1
				supertrend[i] = final_ub[i]

	return supertrend, trend


def compute_supertrend(df, length=10, factor=3.0):
	df = df.copy()
	length = max(1, int(length))
	factor = float(factor)
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=length).average_true_range()
	hl2 = (df["high"] + df["low"]) / 2.0

	basic_ub = hl2 + factor * atr
	basic_lb = hl2 - factor * atr

	# Use optimized NumPy-based calculation
	close_arr = df["close"].to_numpy()
	high_arr = df["high"].to_numpy()
	low_arr = df["low"].to_numpy()
	basic_ub_arr = basic_ub.to_numpy()
	basic_lb_arr = basic_lb.to_numpy()

	supertrend_arr, trend_arr = _compute_supertrend_numba(
		close_arr, high_arr, low_arr, basic_ub_arr, basic_lb_arr
	)

	df["supertrend"] = supertrend_arr
	df["st_trend"] = trend_arr
	df["atr"] = atr
	df["indicator_line"] = df["supertrend"]
	df["trend_flag"] = df["st_trend"]
	return df


def compute_psar(df, step=0.02, max_step=0.2):
	df = df.copy()
	step = float(step)
	max_step = float(max_step)
	high = df["high"].values
	low = df["low"].values
	psar_vals = np.zeros(len(df))
	trend_flags = np.ones(len(df))
	af = step
	ep = high[0]
	psar_vals[0] = low[0]
	bullish = True

	for i in range(1, len(df)):
		prior_psar = psar_vals[i - 1]
		if bullish:
			psar_candidate = prior_psar + af * (ep - prior_psar)
			if i >= 2:
				psar_candidate = min(psar_candidate, low[i - 1], low[i - 2])
			else:
				psar_candidate = min(psar_candidate, low[i - 1])
			if low[i] < psar_candidate:
				bullish = False
				psar_vals[i] = ep
				ep = low[i]
				af = step
			else:
				psar_vals[i] = psar_candidate
				if high[i] > ep:
					ep = high[i]
					af = min(af + step, max_step)
		else:
			psar_candidate = prior_psar + af * (ep - prior_psar)
			if i >= 2:
				psar_candidate = max(psar_candidate, high[i - 1], high[i - 2])
			else:
				psar_candidate = max(psar_candidate, high[i - 1])
			if high[i] > psar_candidate:
				bullish = True
				psar_vals[i] = ep
				ep = high[i]
				af = step
			else:
				psar_vals[i] = psar_candidate
				if low[i] < ep:
					ep = low[i]
					af = min(af + step, max_step)
		trend_flags[i] = 1 if bullish else -1

	df["psar"] = pd.Series(psar_vals, index=df.index)
	df["psar_trend"] = pd.Series(trend_flags, index=df.index)
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["psar"]
	df["trend_flag"] = df["psar_trend"]
	return df


def _compute_jma_numpy(prices: np.ndarray, alpha: float, beta: float, phase_ratio: float) -> np.ndarray:
	"""Optimized JMA calculation using NumPy arrays.

	Avoids pandas indexing overhead for ~5x speedup.
	"""
	n = len(prices)
	jma_values = np.empty(n)
	e0 = prices[0]
	e1 = 0.0
	e2 = 0.0

	for i in range(n):
		price = prices[i]
		e0 = (1.0 - alpha) * price + alpha * e0
		e1 = price - e0
		e2 = (1.0 - beta) * e1 + beta * e2
		jma_values[i] = e0 + phase_ratio * e2

	return jma_values


def jurik_moving_average(series: pd.Series, length: int, phase: int) -> pd.Series:
	if series.empty:
		return pd.Series(index=series.index, dtype=float)
	length = max(1, int(length))
	phase = int(np.clip(phase, -100, 100))
	beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2) if length > 1 else 0.0
	alpha = beta ** 2
	phase_ratio = (phase + 100) / 200

	# Use optimized NumPy calculation
	prices = series.to_numpy()
	jma_arr = _compute_jma_numpy(prices, alpha, beta, phase_ratio)

	return pd.Series(jma_arr, index=series.index)


def compute_jma(df, length=20, phase=0):
	df = df.copy()
	jma = jurik_moving_average(df["close"], length=length, phase=phase)
	trend = np.where(df["close"] >= jma, 1, -1)
	df["jma"] = jma
	df["jma_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["jma"]
	df["trend_flag"] = df["jma_trend"]
	return df


def _compute_kama_numpy(close_arr: np.ndarray, sc_arr: np.ndarray) -> np.ndarray:
	"""Optimized KAMA calculation using NumPy arrays.

	Avoids pandas indexing overhead for ~5x speedup.
	"""
	n = len(close_arr)
	kama = np.empty(n)
	kama[0] = close_arr[0]

	for i in range(1, n):
		kama[i] = kama[i - 1] + sc_arr[i] * (close_arr[i] - kama[i - 1])

	return kama


def compute_kama(df, length=10, slow_length=30, fast_length=2):
	df = df.copy()
	close = df["close"].astype(float)
	length = max(1, int(length))
	slow_length = max(length + 1, int(slow_length))
	fast_length = max(1, int(fast_length))
	fast_sc = 2.0 / (fast_length + 1)
	slow_sc = 2.0 / (slow_length + 1)
	direction = close.diff(length).abs()
	volatility = close.diff().abs().rolling(length).sum()
	er = (direction / volatility).fillna(0.0).clip(lower=0.0, upper=1.0)
	sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

	# Use optimized NumPy calculation
	close_arr = close.to_numpy()
	sc_arr = sc.to_numpy()
	kama_arr = _compute_kama_numpy(close_arr, sc_arr)

	trend = np.where(close_arr >= kama_arr, 1, -1)
	df["kama"] = kama_arr
	df["kama_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["kama"]
	df["trend_flag"] = df["kama_trend"]
	return df


def mesa_adaptive_moving_average(series: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05):
	if series.empty:
		return pd.Series(dtype=float), pd.Series(dtype=float)
	values = series.astype(float)
	n = len(values)
	mama = np.zeros(n)
	fama = np.zeros(n)
	period = np.ones(n) * 10.0
	smooth = values.copy().to_numpy()
	detrender = np.zeros(n)
	I1 = np.zeros(n)
	Q1 = np.zeros(n)
	jI = np.zeros(n)
	jQ = np.zeros(n)
	I2 = np.zeros(n)
	Q2 = np.zeros(n)
	Re = np.zeros(n)
	Im = np.zeros(n)
	phase = np.zeros(n)
	fast_limit = float(max(0.01, fast_limit))
	slow_limit = float(max(0.001, min(fast_limit, slow_limit)))
	for i in range(n):
		price = values.iloc[i]
		if i >= 3:
			smooth[i] = (4 * price + 3 * values.iloc[i - 1] + 2 * values.iloc[i - 2] + values.iloc[i - 3]) / 10.0
		else:
			smooth[i] = price
		if i >= 6:
			detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2] - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * (0.075 * period[i - 1] + 0.54)
			Q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2] - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6]) * (0.075 * period[i - 1] + 0.54)
			I1[i] = detrender[i - 3]
			jI[i] = (0.0962 * I1[i] + 0.5769 * I1[i - 2] - 0.5769 * I1[i - 4] - 0.0962 * I1[i - 6]) * (0.075 * period[i - 1] + 0.54)
			jQ[i] = (0.0962 * Q1[i] + 0.5769 * Q1[i - 2] - 0.5769 * Q1[i - 4] - 0.0962 * Q1[i - 6]) * (0.075 * period[i - 1] + 0.54)
		else:
			detrender[i] = 0.0
			Q1[i] = 0.0
			I1[i] = 0.0
			jI[i] = 0.0
			jQ[i] = 0.0
		I2[i] = I1[i] - jQ[i]
		Q2[i] = Q1[i] + jI[i]
		if i > 0:
			Re[i] = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]
			Im[i] = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]
			angle = np.arctan2(Im[i], Re[i])
			if angle != 0.0:
				period[i] = abs(2 * np.pi / angle)
			period[i] = np.clip(period[i], 6.0, 50.0)
			period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
		else:
			period[i] = period[i - 1] if i > 0 else 10.0
		if I1[i] != 0.0:
			phase[i] = np.degrees(np.arctan2(Q1[i], I1[i]))
		else:
			phase[i] = 0.0
		delta_phase = phase[i - 1] - phase[i] if i > 0 else 0.0
		if delta_phase < 1.0:
			delta_phase = 1.0
		alpha = fast_limit / delta_phase
		alpha = np.clip(alpha, slow_limit, fast_limit)
		mama[i] = alpha * price + (1 - alpha) * (mama[i - 1] if i > 0 else price)
		fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * (fama[i - 1] if i > 0 else price)
	return pd.Series(mama, index=series.index), pd.Series(fama, index=series.index)


def compute_mama(df, fast_limit=0.5, slow_limit=0.05):
	df = df.copy()
	mama, fama = mesa_adaptive_moving_average(df["close"], fast_limit=fast_limit, slow_limit=slow_limit)
	trend = np.where(df["close"] >= mama, 1, -1)
	df["mama"] = mama
	df["fama"] = fama
	df["mama_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["mama"]
	df["trend_flag"] = df["mama_trend"]
	return df


def compute_indicator(df, param_a, param_b, use_cache: bool = True):
	"""Compute the current indicator with optional caching for backtesting performance.

	Args:
		df: Input DataFrame with OHLCV data
		param_a: First indicator parameter
		param_b: Second indicator parameter
		use_cache: Whether to use the indicator cache (default True)

	Returns:
		DataFrame with indicator columns added
	"""
	# Try to retrieve from cache first
	if use_cache:
		df_hash = _hash_dataframe(df)
		cache_key = _get_indicator_cache_key(INDICATOR_TYPE, df_hash, param_a, param_b)

		if cache_key in INDICATOR_CACHE:
			# Return a copy to prevent mutation of cached data
			return INDICATOR_CACHE[cache_key].copy()

	# Compute the indicator
	if INDICATOR_TYPE == "supertrend":
		result = compute_supertrend(df, length=int(param_a), factor=float(param_b))
	elif INDICATOR_TYPE == "psar":
		result = compute_psar(df, step=float(param_a), max_step=float(param_b))
	elif INDICATOR_TYPE == "jma":
		result = compute_jma(df, length=int(param_a), phase=int(param_b))
	elif INDICATOR_TYPE == "kama":
		result = compute_kama(df, length=int(param_a), slow_length=int(param_b))
	elif INDICATOR_TYPE == "mama":
		result = compute_mama(df, fast_limit=float(param_a), slow_limit=float(param_b))
	else:
		raise ValueError(f"Unsupported INDICATOR_TYPE: {INDICATOR_TYPE}")

	# Store in cache if enabled (with size limit)
	if use_cache:
		if len(INDICATOR_CACHE) >= INDICATOR_CACHE_MAX_SIZE:
			# Remove oldest entry (first key)
			try:
				oldest_key = next(iter(INDICATOR_CACHE))
				del INDICATOR_CACHE[oldest_key]
			except StopIteration:
				pass
		INDICATOR_CACHE[cache_key] = result.copy()

	return result


def attach_higher_timeframe_trend(df_low, symbol):
	if not USE_HIGHER_TIMEFRAME_FILTER:
		df_low = df_low.copy()
		df_low["htf_trend"] = 0
		df_low["htf_indicator"] = np.nan
		return df_low

	df_high = fetch_data(symbol, HIGHER_TIMEFRAME, HTF_LOOKBACK)
	if df_high.empty:
		df_low = df_low.copy()
		df_low["htf_trend"] = 0
		df_low["htf_indicator"] = np.nan
		return df_low

	if INDICATOR_TYPE == "supertrend":
		df_high_ind = compute_supertrend(df_high, length=HTF_LENGTH, factor=HTF_FACTOR)
		indicator_col = "supertrend"
		trend_col = "st_trend"
	elif INDICATOR_TYPE == "psar":
		df_high_ind = compute_psar(df_high, step=HTF_PSAR_STEP, max_step=HTF_PSAR_MAX_STEP)
		indicator_col = "psar"
		trend_col = "psar_trend"
	elif INDICATOR_TYPE == "jma":
		df_high_ind = compute_jma(df_high, length=HTF_JMA_LENGTH, phase=HTF_JMA_PHASE)
		indicator_col = "jma"
		trend_col = "jma_trend"
	elif INDICATOR_TYPE == "kama":
		df_high_ind = compute_kama(df_high, length=HTF_KAMA_LENGTH, slow_length=HTF_KAMA_SLOW_LENGTH)
		indicator_col = "kama"
		trend_col = "kama_trend"
	elif INDICATOR_TYPE == "mama":
		df_high_ind = compute_mama(df_high, fast_limit=HTF_MAMA_FAST_LIMIT, slow_limit=HTF_MAMA_SLOW_LIMIT)
		indicator_col = "mama"
		trend_col = "mama_trend"
	else:
		raise ValueError(f"Unsupported HTF indicator type: {INDICATOR_TYPE}")

	htf = df_high_ind[[indicator_col, trend_col]].rename(columns={
		indicator_col: "htf_indicator",
		trend_col: "htf_trend"
	})
	aligned = htf.reindex(df_low.index, method="ffill")
	df_low = df_low.copy()
	df_low["htf_trend"] = aligned["htf_trend"].fillna(0).astype(int)
	df_low["htf_indicator"] = aligned["htf_indicator"]
	return df_low


def attach_momentum_filter(df):
	df = df.copy()
	if not USE_MOMENTUM_FILTER:
		df["momentum"] = np.nan
		return df

	if MOMENTUM_TYPE.lower() == "rsi":
		delta = df["close"].diff()
		gain = np.where(delta > 0, delta, 0.0)
		loss = np.where(delta < 0, -delta, 0.0)
		roll_gain = pd.Series(gain, index=df.index).rolling(MOMENTUM_WINDOW).mean()
		roll_loss = pd.Series(loss, index=df.index).rolling(MOMENTUM_WINDOW).mean()
		rs = roll_gain / roll_loss.replace(0, np.nan)
		rsi = 100 - (100 / (1 + rs))
		df["momentum"] = rsi
	else:
		df["momentum"] = np.nan
	return df


def prepare_symbol_dataframe(symbol):
	df = fetch_data(symbol, TIMEFRAME, LOOKBACK)
	df = attach_higher_timeframe_trend(df, symbol)
	df = attach_momentum_filter(df)
	return df


def backtest_supertrend(df, atr_stop_mult=None, direction="long", min_hold_bars=0, min_hold_days=None):
	direction = direction.lower()
	if direction not in {"long", "short"}:
		raise ValueError("direction must be 'long' or 'short'")
	min_hold_bars = 0 if min_hold_bars is None else max(0, int(min_hold_bars))
	min_hold_days = min_hold_days if min_hold_days is not None else 0

	long_mode = direction == "long"
	equity = START_EQUITY
	trades = []
	in_position = False
	entry_price = None
	entry_ts = None
	entry_capital = None
	entry_atr = None
	bars_in_position = 0

	for i in range(1, len(df)):
		ts = df.index[i]
		trend = int(df["trend_flag"].iloc[i])
		prev_trend = int(df["trend_flag"].iloc[i - 1])

		enter_long = prev_trend == -1 and trend == 1
		enter_short = prev_trend == 1 and trend == -1

		if not in_position:
			htf_value = int(df["htf_trend"].iloc[i]) if "htf_trend" in df.columns else 0
			htf_allows = True
			if USE_HIGHER_TIMEFRAME_FILTER:
				htf_allows = htf_value >= 1 if long_mode else htf_value <= -1

			momentum_allows = True
			if USE_MOMENTUM_FILTER and "momentum" in df.columns:
				mom_value = df["momentum"].iloc[i]
				if pd.isna(mom_value):
					momentum_allows = False
				else:
					momentum_allows = mom_value >= RSI_LONG_THRESHOLD if long_mode else mom_value <= RSI_SHORT_THRESHOLD

			breakout_allows = True
			if USE_BREAKOUT_FILTER:
				atr_curr = df["atr"].iloc[i]
				if atr_curr is None or np.isnan(atr_curr) or atr_curr <= 0:
					breakout_allows = False
				else:
					candle_range = float(df["high"].iloc[i] - df["low"].iloc[i])
					breakout_allows = candle_range >= BREAKOUT_ATR_MULT * float(atr_curr)
					if breakout_allows and BREAKOUT_REQUIRE_DIRECTION:
						prev_high = float(df["high"].iloc[i - 1]) if i > 0 else float(df["high"].iloc[i])
						prev_low = float(df["low"].iloc[i - 1]) if i > 0 else float(df["low"].iloc[i])
						close_curr = float(df["close"].iloc[i])
						breakout_allows = close_curr > prev_high if long_mode else close_curr < prev_low

			if long_mode and enter_long and htf_allows and momentum_allows and breakout_allows:
				in_position = True
			elif (not long_mode) and enter_short and htf_allows and momentum_allows and breakout_allows:
				in_position = True

			if in_position:
				entry_price = float(df["close"].iloc[i])
				entry_ts = ts
				entry_capital = equity / STAKE_DIVISOR
				atr_val = df["atr"].iloc[i]
				entry_atr = float(atr_val) if not np.isnan(atr_val) else 0.0
				bars_in_position = 0
			continue

		bars_in_position += 1
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		atr_buffer = entry_atr if entry_atr is not None else 0.0
		stop_price = None
		if atr_stop_mult is not None and atr_buffer and atr_buffer > 0:
			stop_price = entry_price - atr_stop_mult * atr_buffer if long_mode else entry_price + atr_stop_mult * atr_buffer

		exit_price = None
		exit_reason = None

		if stop_price is not None:
			if long_mode and float(df["low"].iloc[i]) <= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"
			elif (not long_mode) and float(df["high"].iloc[i]) >= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"

		if exit_price is None:
			if long_mode and prev_trend == 1 and trend == -1 and bars_in_position >= min_hold_bars:
				exit_price = float(df["close"].iloc[i])
				exit_reason = "Trend flip"
			elif (not long_mode) and prev_trend == -1 and trend == 1 and bars_in_position >= min_hold_bars:
				exit_price = float(df["close"].iloc[i])
				exit_reason = "Trend flip"

		if exit_price is None:
			continue

		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		gross_pnl = price_diff / entry_price * stake
		fees = stake * FEE_RATE * 2.0
		pnl_usd = gross_pnl - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": ts,
			"ExitPreis": exit_price,
			"Stake": stake,
			"Fees": fees,
			"ExitReason": exit_reason,
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldDays": min_hold_days
		})
		in_position = False
		entry_capital = None
		entry_atr = None
		bars_in_position = 0

	if in_position:
		last = df.iloc[-1]
		exit_ts = last.name
		exit_price = float(last["close"])
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		gross_pnl = price_diff / entry_price * stake
		fees = stake * FEE_RATE * 2.0
		pnl_usd = gross_pnl - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": exit_ts,
			"ExitPreis": exit_price,
			"Stake": stake,
			"Fees": fees,
			"ExitReason": "Final bar",
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldDays": min_hold_days
		})

	return pd.DataFrame(trades)


def performance_report(trades_df, symbol, param_a, param_b, direction, min_hold_days):
	base = {
		"Symbol": symbol,
		"ParamA": param_a,
		"ParamB": param_b,
		PARAM_A_LABEL: param_a,
		PARAM_B_LABEL: param_b,
	}
	if INDICATOR_TYPE == "supertrend":
		base["Length"] = param_a
		base["Factor"] = param_b

	if trades_df.empty:
		return {
			**base,
			"Trades": 0,
			"WinRate": 0.0,
			"AvgPnL": 0.0,
			"ProfitFactor": 0.0,
			"MaxDrawdown": 0.0,
			"FinalEquity": START_EQUITY,
			"Direction": direction,
			"MinHoldDays": min_hold_days,
		}

	wins = trades_df[trades_df["PnL (USD)"] > 0]
	losses = trades_df[trades_df["PnL (USD)"] < 0]
	win_rate = len(wins) / len(trades_df)
	avg_pnl = trades_df["PnL (USD)"].mean()
	total_win = wins["PnL (USD)"].sum()
	total_loss = abs(losses["PnL (USD)"].sum())
	profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
	equity_curve = trades_df["Equity"]
	max_drawdown = (equity_curve.cummax() - equity_curve).max() if not equity_curve.empty else 0.0
	final_eq = float(equity_curve.iloc[-1]) if not equity_curve.empty else START_EQUITY
	return {
		**base,
		"Trades": len(trades_df),
		"WinRate": win_rate,
		"AvgPnL": avg_pnl,
		"ProfitFactor": profit_factor,
		"MaxDrawdown": max_drawdown,
		"FinalEquity": final_eq,
		"Direction": direction,
		"MinHoldDays": min_hold_days,
	}


def build_equity_series(df, trades_df, direction):
	if df.empty:
		return pd.Series(dtype=float)
	equity_series = pd.Series(index=df.index, dtype=float)
	equity_series.iloc[0] = START_EQUITY
	if trades_df.empty:
		return equity_series.ffill()
	current_equity = START_EQUITY
	direction_key = str(direction).lower()
	for trade in trades_df.to_dict("records"):
		entry_ts = trade.get("Zeit")
		exit_ts = trade.get("ExitZeit")
		stake = float(trade.get("Stake", START_EQUITY / STAKE_DIVISOR))
		entry_price = float(trade.get("Entry", 0))
		if not entry_ts or not exit_ts or entry_price == 0:
			continue
		entry_idx = df.index.get_indexer([entry_ts])
		exit_idx = df.index.get_indexer([exit_ts])
		if entry_idx.size == 0 or exit_idx.size == 0:
			continue
		entry_pos = entry_idx[0]
		exit_pos = exit_idx[0]
		if entry_pos == -1 or exit_pos == -1 or exit_pos < entry_pos:
			continue
		price_slice = df.iloc[entry_pos:exit_pos + 1]["close"]
		baseline_equity = current_equity
		for ts, close_price in price_slice.items():
			if ts == exit_ts:
				equity_at_exit = float(trade.get("Equity", baseline_equity))
				equity_series.loc[ts] = equity_at_exit
				current_equity = equity_at_exit
				break
			close_val = float(close_price)
			if direction_key == "short":
				unrealized = (entry_price - close_val) / entry_price * stake
			else:
				unrealized = (close_val - entry_price) / entry_price * stake
			equity_series.loc[ts] = baseline_equity + unrealized
	return equity_series.ffill().fillna(START_EQUITY)


def build_two_panel_figure(symbol, df, trades_df, param_a, param_b, direction, min_hold_days=None):
	direction_title = direction.capitalize()
	hold_text = f", Hold≥{min_hold_days}d" if min_hold_days else ""
	indicator_desc = f"{INDICATOR_DISPLAY_NAME} {PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"
	line_name = "Supertrend" if INDICATOR_TYPE == "supertrend" else INDICATOR_DISPLAY_NAME
	fig = make_subplots(
		rows=3,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.06,
		row_heights=[0.55, 0.3, 0.15],
		subplot_titles=(
			f"{symbol} {direction_title} {indicator_desc}{hold_text}",
			"Equity",
			"Momentum",
		),
	)

	fig.add_trace(
		go.Candlestick(
			x=df.index,
			open=df["open"],
			high=df["high"],
			low=df["low"],
			close=df["close"],
			name="Price",
		),
		row=1,
		col=1,
	)
	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=df["indicator_line"],
			mode="lines",
			name=line_name,
			line=dict(color="orange"),
		),
		row=1,
		col=1,
	)
	if USE_HIGHER_TIMEFRAME_FILTER and "htf_indicator" in df.columns:
		fig.add_trace(
			go.Scatter(
				x=df.index,
				y=df["htf_indicator"],
				mode="lines",
				name=f"HTF {INDICATOR_DISPLAY_NAME} ({HIGHER_TIMEFRAME})",
				line=dict(color="purple", dash="dot"),
			),
			row=1,
			col=1,
		)

	if not trades_df.empty:
		entry_color = "green" if direction_title == "Long" else "red"
		exit_color = "red" if direction_title == "Long" else "green"
		entry_symbol = "triangle-up" if direction_title == "Long" else "triangle-down"
		exit_symbol = "triangle-down" if direction_title == "Long" else "triangle-up"
		fig.add_trace(
			go.Scatter(
				x=trades_df["Zeit"],
				y=trades_df["Entry"],
				mode="markers",
				marker=dict(color=entry_color, symbol=entry_symbol, size=10),
				name=f"{direction_title} Entry",
			),
			row=1,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=trades_df["ExitZeit"],
				y=trades_df["ExitPreis"],
				mode="markers",
				marker=dict(color=exit_color, symbol=exit_symbol, size=10),
				name=f"{direction_title} Exit",
			),
			row=1,
			col=1,
		)

		equity_series = build_equity_series(df, trades_df, direction_title)
		fig.add_trace(
			go.Scatter(x=equity_series.index, y=equity_series, mode="lines", name="Equity"),
			row=2,
			col=1,
		)
	else:
		equity_series = pd.Series(index=df.index, data=START_EQUITY, dtype=float) if len(df.index) else None
		if equity_series is not None:
			fig.add_trace(
				go.Scatter(x=equity_series.index, y=equity_series, mode="lines", name="Equity"),
				row=2,
				col=1,
			)

	if "momentum" in df.columns:
		fig.add_trace(
			go.Scatter(x=df.index, y=df["momentum"], mode="lines", name="Momentum", line=dict(color="teal")),
			row=3,
			col=1,
		)
		fig.add_hrect(
			y0=RSI_SHORT_THRESHOLD,
			y1=RSI_LONG_THRESHOLD,
			line_width=0,
			fillcolor="gray",
			opacity=0.15,
			row=3,
			col=1,
		)

	fig.update_layout(
		height=900,
		showlegend=True,
		xaxis=dict(rangeslider=dict(visible=False)),
		xaxis2=dict(rangeslider=dict(visible=False)),
		xaxis3=dict(rangeslider=dict(visible=True, thickness=0.03), type="date"),
	)
	fig.update_xaxes(title_text="Zeit", row=3, col=1)
	fig.update_yaxes(title_text="Preis", row=1, col=1)
	fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)
	fig.update_yaxes(title_text="RSI", row=3, col=1)
	return fig


def df_to_html_table(df, title=None):
	html = ""
	if title:
		html += f"<h3>{title}</h3>\n"
	if df.empty:
		html += "<p>Keine Daten</p>"
	else:
		html += df.to_html(index=False, justify="left", border=0)
	return html


def _format_result_cell(entry):
	if not entry:
		return "-"
	win_pct = float(entry.get("WinRate", 0.0)) * 100.0
	max_dd = float(entry.get("MaxDrawdown", 0.0))
	return (
		f"HTF {entry.get('HTF', '-')}, Eq {entry.get('FinalEquity', START_EQUITY):.0f}, "
		f"Trades {entry.get('Trades', 0)}, Win {win_pct:.1f}%, DD {max_dd:.0f}"
	)


def record_global_best(indicator_key, summary_rows):
	if not summary_rows:
		return
	indicator_store = GLOBAL_BEST_RESULTS.setdefault(indicator_key, {})
	for row in summary_rows:
		symbol = row.get("Symbol")
		direction = str(row.get("Direction", "Long")).lower()
		if direction not in {"long", "short"} or not symbol:
			continue
		symbol_store = indicator_store.setdefault(symbol, {})
		existing = symbol_store.get(direction)
		candidate_equity = float(row.get("FinalEquity", START_EQUITY))
		existing_equity = float(existing.get("FinalEquity", START_EQUITY)) if existing else None
		if existing is None or candidate_equity > existing_equity:
			symbol_store[direction] = dict(row)


def write_overall_result_tables():
	if not GLOBAL_BEST_RESULTS:
		return
	indicator_order = list(INDICATOR_PRESETS.keys())
	indicator_labels = {key: INDICATOR_PRESETS[key]["display_name"] for key in indicator_order}
	long_rows = []
	short_rows = []
	for symbol in SYMBOLS:
		long_row = {"Symbol": symbol}
		short_row = {"Symbol": symbol}
		for key in indicator_order:
			entry_long = GLOBAL_BEST_RESULTS.get(key, {}).get(symbol, {}).get("long")
			entry_short = GLOBAL_BEST_RESULTS.get(key, {}).get(symbol, {}).get("short")
			col_name = indicator_labels[key]
			long_row[col_name] = _format_result_cell(entry_long)
			short_row[col_name] = _format_result_cell(entry_short)
		long_rows.append(long_row) 
		short_rows.append(short_row)
	long_df = pd.DataFrame(long_rows)
	short_df = pd.DataFrame(short_rows)
	os.makedirs(BASE_OUT_DIR, exist_ok=True)
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html_parts = [
		"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Overall Indicator Results</title></head><body>",
		f"<h1>Overall Indicator Results</h1><p>Stand: {now}</p>",
		"<h2>Long Ergebnisse</h2>",
		long_df.to_html(index=False, justify="left", border=0) if not long_df.empty else "<p>Keine Daten</p>",
		"<h2>Short Ergebnisse</h2>",
		short_df.to_html(index=False, justify="left", border=0) if not short_df.empty else "<p>Keine Daten</p>",
		"</body></html>",
	]
	with open(OVERALL_SUMMARY_HTML, "w", encoding="utf-8") as f:
		f.write("\n".join(html_parts))
	csv_rows = []
	for key in indicator_order:
		indicator_store = GLOBAL_BEST_RESULTS.get(key, {})
		for symbol, dir_dict in indicator_store.items():
			for direction, entry in dir_dict.items():
				row = dict(entry)
				row["Indicator"] = key
				row["IndicatorDisplay"] = indicator_labels[key]
				row["Direction"] = direction.capitalize()
				csv_rows.append(row)
	if csv_rows:
		pd.DataFrame(csv_rows).to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")


def write_combined_overall_best_report(sections):
	if not sections:
		return
	best_per_symbol = {}
	for item in sections:
		symbol = item.get("symbol")
		if not symbol:
			continue
		value = item.get("final_equity")
		try:
			value = float(value)
		except (TypeError, ValueError):
			value = float("-inf")
		current = best_per_symbol.get(symbol)
		if current is None or value > current[0]:
			best_per_symbol[symbol] = (value, item)
	sections = [entry for (_, entry) in best_per_symbol.values()]
	sections.sort(key=lambda item: item.get("symbol", ""))
	long_entries = [s for s in sections if s.get("direction", "").lower() == "long"]
	short_entries = [s for s in sections if s.get("direction", "").lower() == "short"]
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html = [
		"<!DOCTYPE html>",
		"<html><head><meta charset='utf-8'><title>Overall-Best Detailreport</title>",
		"<style>body{font-family:Arial, sans-serif;margin:20px;}h2{margin-top:40px;}section{margin-bottom:50px;}table{margin-top:10px;}hr{margin:40px 0;}</style>",
		"</head><body>",
		f"<h1>Overall-Best Detailreport</h1><p>Stand: {now}</p>",
	]

	def render_entry(entry):
		meta = (
			f"<h3>{entry['symbol']} – {entry['direction']} – {entry['indicator']} ({entry['htf']})<br>"
			f"{entry['param_desc']}, ATRStop={entry['atr_label']}, MinHold={entry['min_hold_days']}d</h3>"
		)
		html.append("<section>")
		html.append(meta)
		html.append(entry["fig_html"])
		html.append(entry["trade_table_html"])
		html.append("</section>")

	if long_entries:
		html.append("<h2>Long Trades</h2>")
		for item in long_entries:
			render_entry(item)
	if short_entries:
		html.append("<h2>Short Trades</h2>")
		for item in short_entries:
			render_entry(item)

	html.append("</body></html>")
	with open(OVERALL_DETAILED_HTML, "w", encoding="utf-8") as f:
		f.write("\n".join(html))


def write_flat_trade_list(rows):
	os.makedirs(BASE_OUT_DIR, exist_ok=True)
	flat_columns = [
		"Indicator",
		"IndicatorDisplay",
		"HTF",
		"Symbol",
		"Direction",
		"ParamA",
		"ParamB",
		PARAM_A_LABEL,
		PARAM_B_LABEL,
		"ATRStopMultValue",
		"ATRStopMult",
		"MinHoldDays",
		"FinalEquity",
		"Trades",
		"WinRate",
		"MaxDrawdown",
		"TradesCSV",
	]
	if not rows:
		flat_df = pd.DataFrame(columns=flat_columns)
	else:
		flat_df = pd.DataFrame(rows)
		for col in flat_columns:
			if col not in flat_df.columns:
				flat_df[col] = ""
		flat_df = flat_df.sort_values(["Indicator", "HTF", "Symbol", "Direction"]).reset_index(drop=True)
	flat_df = flat_df.reindex(columns=flat_columns)
	csv_path = OVERALL_FLAT_CSV
	json_path = OVERALL_FLAT_JSON
	try:
		flat_df.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
	except PermissionError as exc:
		timestamp = datetime.now(BERLIN_TZ).strftime("%Y%m%d_%H%M%S")
		csv_path = os.path.join(BASE_OUT_DIR, f"overall_best_flat_trades_{timestamp}.csv")
		flat_df.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
		print(f"[Flat] {exc}. Wrote fallback CSV {csv_path}")
	try:
		records = json.loads(flat_df.to_json(orient="records"))
		with open(json_path, "w", encoding="utf-8") as fh:
			json.dump(records, fh, ensure_ascii=False, indent=2)
	except PermissionError as exc:
		timestamp = datetime.now(BERLIN_TZ).strftime("%Y%m%d_%H%M%S")
		json_path = os.path.join(BASE_OUT_DIR, f"overall_best_flat_trades_{timestamp}.json")
		with open(json_path, "w", encoding="utf-8") as fh:
			json.dump(records, fh, ensure_ascii=False, indent=2)
		print(f"[Flat] {exc}. Wrote fallback JSON {json_path}")
	print(f"[Flat] Saved {len(flat_df)} trade definitions to {csv_path}")


def build_full_report(figs_html_blocks, sections_html, ranking_tables_html):
	html = []
	page_title = f"{INDICATOR_DISPLAY_NAME} Parameter Report ({HIGHER_TIMEFRAME})"
	html.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{page_title}</title></head><body>")
	html.append(f"<h1>{page_title}</h1>")
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html.append(f"<p>Generiert: {now}</p>")

	for blk in figs_html_blocks:
		html.append(blk)
	for sec in sections_html:
		html.append(sec)

	html.append("<hr>")
	html.append("<h2>Parameter-Ranking je Symbol</h2>")
	for rtbl in ranking_tables_html:
		html.append(rtbl)

	html.append("</body></html>")
	return "\n".join(html)


def _normalize_param_values(param_a, param_b):
	if pd.isna(param_a):
		param_a = DEFAULT_PARAM_A
	if pd.isna(param_b):
		param_b = DEFAULT_PARAM_B
	if INDICATOR_TYPE in {"supertrend", "jma", "kama"}:
		param_a = int(round(float(param_a)))
	else:
		param_a = float(param_a)
	if INDICATOR_TYPE in {}:
		param_b = int(round(float(param_b)))
	else:
		param_b = float(param_b)
	return param_a, param_b


def _normalize_atr_value(raw_value):
	if isinstance(raw_value, str):
		raw_str = raw_value.strip().lower()
		if not raw_str or raw_str == "none":
			return None
		return float(raw_str)
	if pd.isna(raw_value):
		return None
	return float(raw_value)


def _run_saved_rows(rows_df, table_title, save_path=None, aggregate_sections=None):
	if rows_df is None or rows_df.empty:
		print("[Skip] No saved parameter rows to execute.")
		return []
	rows_df = rows_df.copy()
	print(f"[Run] {table_title} – {len(rows_df)} gespeicherte Kombinationen")
	figs_blocks = []
	sections_blocks = []
	ranking_tables = []
	data_cache = {}
	st_cache = {}
	updated_rows = []
	for _, row in rows_df.iterrows():
		row_dict = row.to_dict()
		symbol = row.get("Symbol")
		if not symbol:
			continue
		direction = str(row.get("Direction", "Long")).lower()
		if direction not in {"long", "short"}:
			continue
		if direction == "long" and not ENABLE_LONGS:
			continue
		if direction == "short" and not ENABLE_SHORTS:
			continue
		param_a = row.get("ParamA", DEFAULT_PARAM_A)
		param_b = row.get("ParamB", DEFAULT_PARAM_B)
		param_a, param_b = _normalize_param_values(param_a, param_b)
		atr_mult = row.get("ATRStopMultValue", row.get("ATRStopMult"))
		atr_mult = _normalize_atr_value(atr_mult)
		hold_days = row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS)
		if pd.isna(hold_days):
			hold_days = DEFAULT_MIN_HOLD_DAYS
		else:
			hold_days = int(hold_days)
		min_hold_bars = hold_days * BARS_PER_DAY
		if symbol not in data_cache:
			data_cache[symbol] = prepare_symbol_dataframe(symbol)
		df_raw = data_cache[symbol]
		st_key = (symbol, param_a, param_b)
		if st_key not in st_cache:
			df_tmp = compute_indicator(df_raw, param_a, param_b)
			for col in ("htf_trend", "htf_indicator", "momentum"):
				if col in df_raw.columns:
					df_tmp[col] = df_raw[col]
			st_cache[st_key] = df_tmp
		df_st = st_cache[st_key]
		trades = backtest_supertrend(
			df_st,
			atr_stop_mult=atr_mult,
			direction=direction,
			min_hold_bars=min_hold_bars,
			min_hold_days=hold_days,
		)
		direction_title = direction.capitalize()
		atr_label = "None" if atr_mult is None else atr_mult
		param_desc = f"{PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"
		print(f"  · {symbol} {direction_title} ({param_desc}, ATR={atr_label}, MinHold={hold_days}d)")
		stats = performance_report(
			trades,
			symbol,
			param_a,
			param_b,
			direction_title,
			hold_days,
		)
		updated_row = dict(row_dict)
		updated_row.update({
			"ParamA": param_a,
			"ParamB": param_b,
			PARAM_A_LABEL: param_a,
			PARAM_B_LABEL: param_b,
			"MinHoldDays": hold_days,
			"ATRStopMult": atr_label,
			"ATRStopMultValue": atr_mult,
			"HTF": HIGHER_TIMEFRAME,
			"Trades": stats["Trades"],
			"WinRate": stats["WinRate"],
			"AvgPnL": stats["AvgPnL"],
			"ProfitFactor": stats["ProfitFactor"],
			"MaxDrawdown": stats["MaxDrawdown"],
			"FinalEquity": stats["FinalEquity"],
		})
		updated_rows.append(updated_row)
		fig = build_two_panel_figure(
			symbol,
			df_st,
			trades,
			param_a,
			param_b,
			direction_title,
			min_hold_days=hold_days,
		)
		fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
		figs_blocks.append(
			f"<h2>{symbol} – {direction_title} gespeicherte Parameter: {param_desc}, ATRStop={atr_label}, MinHold={hold_days}d</h2>\n"
			+ fig_html
		)
		trade_table_html = df_to_html_table(
			trades,
			title=f"Trade-Liste {symbol} ({direction_title} gespeicherte Parameter, MinHold={hold_days}d)",
		)
		sections_blocks.append(trade_table_html)
		csv_suffix = "" if direction == "long" else "_short"
		csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
		trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
		updated_row["TradesCSV"] = csv_path
		if aggregate_sections is not None:
			aggregate_sections.append({
				"indicator": INDICATOR_DISPLAY_NAME,
				"indicator_slug": INDICATOR_SLUG,
				"htf": HIGHER_TIMEFRAME,
				"symbol": symbol,
				"direction": direction_title,
				"param_desc": param_desc,
				"atr_label": atr_label,
				"min_hold_days": hold_days,
				"fig_html": fig_html,
				"trade_table_html": trade_table_html,
				"final_equity": stats.get("FinalEquity", START_EQUITY),
			})
	updated_df = pd.DataFrame(updated_rows) if updated_rows else rows_df
	if updated_df is not None:
		ranking_tables = [df_to_html_table(updated_df, title=table_title)]
	if figs_blocks or sections_blocks:
		report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
		report_path = os.path.join(OUT_DIR, REPORT_FILE)
		with open(report_path, "w", encoding="utf-8") as f:
			f.write(report_html)
	if save_path and updated_df is not None and not updated_df.empty:
		dir_name = os.path.dirname(save_path) or "."
		os.makedirs(dir_name, exist_ok=True)
		updated_df.to_csv(save_path, sep=";", decimal=",", index=False, encoding="utf-8")
	return updated_df.to_dict("records") if updated_df is not None else []


def run_parameter_sweep():
	figs_blocks = []
	sections_blocks = []
	ranking_tables = []
	best_params_summary = []
	clear_directory(OUT_DIR)

	directions = get_enabled_directions()
	hold_day_candidates = MIN_HOLD_DAY_VALUES if USE_MIN_HOLD_FILTER else [DEFAULT_MIN_HOLD_DAYS]

	for symbol in SYMBOLS:
		df_raw = prepare_symbol_dataframe(symbol)
		results = {d: [] for d in directions}
		trades_per_combo = {d: {} for d in directions}
		df_cache = {}

		for param_a in PARAM_A_VALUES:
			for param_b in PARAM_B_VALUES:
				cache_key = (param_a, param_b)
				if cache_key not in df_cache:
					df_tmp = compute_indicator(df_raw, param_a, param_b)
					for col in ("htf_trend", "htf_indicator", "momentum"):
						if col in df_raw.columns:
							df_tmp[col] = df_raw[col]
					df_cache[cache_key] = df_tmp
				df_st = df_cache[cache_key]
				for atr_mult in ATR_STOP_MULTS:
					for hold_days in hold_day_candidates:
						min_hold_bars = hold_days * BARS_PER_DAY
						for direction in directions:
							df_st_with_htf = df_st.copy()
							for col in ("htf_trend", "htf_indicator", "momentum"):
								if col in df_raw.columns:
									df_st_with_htf[col] = df_raw[col]
							trades = backtest_supertrend(
								df_st_with_htf,
								atr_stop_mult=atr_mult,
								direction=direction,
								min_hold_bars=min_hold_bars,
								min_hold_days=hold_days,
							)
							stats = performance_report(
								trades,
								symbol,
								param_a,
								param_b,
								direction.capitalize(),
								hold_days,
							)
							stats["ATRStopMult"] = atr_mult if atr_mult is not None else "None"
							stats["MinHoldBars"] = min_hold_bars
							results[direction].append(stats)
							trades_per_combo[direction][(param_a, param_b, atr_mult, hold_days)] = trades

		for direction in directions:
			dir_results = results[direction]
			ranking_df = pd.DataFrame(dir_results)
			if not ranking_df.empty:
				ranking_df = ranking_df.sort_values("FinalEquity", ascending=False).reset_index(drop=True)
			ranking_tables.append(
				df_to_html_table(
					ranking_df,
					title=f"Ranking: {symbol} {INDICATOR_DISPLAY_NAME} ({direction.capitalize()} nach FinalEquity)",
				)
			)

			best_param_a, best_param_b = DEFAULT_PARAM_A, DEFAULT_PARAM_B
			best_atr = None
			best_hold_days = DEFAULT_MIN_HOLD_DAYS
			final_equity = START_EQUITY
			trades_count = 0
			win_rate = 0.0
			max_dd = 0.0
			if not ranking_df.empty:
				best_row = ranking_df.iloc[0]
				best_param_a = best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
				best_param_b = best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
				best_param_a = best_param_a if not pd.isna(best_param_a) else DEFAULT_PARAM_A
				best_param_b = best_param_b if not pd.isna(best_param_b) else DEFAULT_PARAM_B
				best_atr_raw = best_row.get("ATRStopMult", "None")
				best_atr = best_atr_raw if best_atr_raw != "None" else None
				best_hold_days = int(best_row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS))
				final_equity = float(best_row.get("FinalEquity", START_EQUITY))
				trades_count = int(best_row.get("Trades", 0))
				win_rate = float(best_row.get("WinRate", 0.0))
				max_dd = float(best_row.get("MaxDrawdown", 0.0))
				best_df = df_cache[(best_param_a, best_param_b)]
				best_trades = trades_per_combo[direction][(best_param_a, best_param_b, best_atr, best_hold_days)]
			else:
				best_df = compute_indicator(df_raw, best_param_a, best_param_b)
				for col in ("htf_trend", "htf_indicator", "momentum"):
					if col in df_raw.columns:
						best_df[col] = df_raw[col]
				best_trades = pd.DataFrame()

			atr_label = best_atr if best_atr is not None else "None"
			best_params_summary.append({
				"Symbol": symbol,
				"Direction": direction.capitalize(),
				"Indicator": INDICATOR_TYPE,
				"IndicatorDisplay": INDICATOR_DISPLAY_NAME,
				"ParamA": best_param_a,
				"ParamB": best_param_b,
				PARAM_A_LABEL: best_param_a,
				PARAM_B_LABEL: best_param_b,
				"Length": best_param_a if INDICATOR_TYPE == "supertrend" else None,
				"Factor": best_param_b if INDICATOR_TYPE == "supertrend" else None,
				"ATRStopMult": atr_label,
				"ATRStopMultValue": best_atr,
				"MinHoldDays": best_hold_days,
				"HTF": HIGHER_TIMEFRAME,
				"FinalEquity": final_equity,
				"Trades": trades_count,
				"WinRate": win_rate,
				"MaxDrawdown": max_dd,
			})

			fig = build_two_panel_figure(
				symbol,
				best_df,
				best_trades,
				best_param_a,
				best_param_b,
				direction.capitalize(),
				min_hold_days=best_hold_days,
			)
			fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
			figs_blocks.append(
				f"<h2>{symbol} – {direction.capitalize()} beste Parameter: {PARAM_A_LABEL}={best_param_a}, {PARAM_B_LABEL}={best_param_b}, ATRStop={atr_label}, MinHold={best_hold_days}d</h2>\n"
				+ fig_html
			)

			sections_blocks.append(
				df_to_html_table(
					best_trades,
					title=f"Trade-Liste {symbol} ({direction.capitalize()} beste Parameter, MinHold={best_hold_days}d)",
				)
			)

			csv_suffix = "" if direction == "long" else "_short"
			csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
			best_trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
			csv_rank_suffix = "" if direction == "long" else "_short"
			csv_rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_params{csv_rank_suffix}.csv")
			ranking_df.to_csv(csv_rank_path, sep=";", decimal=",", index=False, encoding="utf-8")

	if best_params_summary:
		summary_df = pd.DataFrame(best_params_summary)
		summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
		summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")

	report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
	report_path = os.path.join(OUT_DIR, REPORT_FILE)
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report_html)
	return best_params_summary


def run_saved_params(rows_df=None):
	summary_df = rows_df
	summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
	if summary_df is None:
		if os.path.exists(summary_path):
			summary_df = pd.read_csv(summary_path, sep=";", decimal=",")
		else:
			summary_path = None
	if summary_df is None or summary_df.empty:
		print("[Skip] No saved parameters available. Run the sweep to generate them.")
		return []
	return _run_saved_rows(
		summary_df,
		table_title="Gespeicherte Parameter (ohne Sweep)",
		save_path=summary_path,
	)


def run_overall_best_params():
	if not os.path.exists(OVERALL_PARAMS_CSV):
		print("[Skip] Overall summary file missing. Run parameter sweeps first.")
		return []
	overall_df = pd.read_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",")
	if overall_df.empty:
		print("[Skip] Overall summary file is empty.")
		return []
	if ACTIVE_INDICATORS:
		overall_df = overall_df[overall_df["Indicator"].isin(ACTIVE_INDICATORS)]
		if overall_df.empty:
			print("[Skip] No overall rows match ACTIVE_INDICATORS.")
			return []
	allowed_dirs = []
	if ENABLE_LONGS:
		allowed_dirs.append("Long")
	if ENABLE_SHORTS:
		allowed_dirs.append("Short")
	if not allowed_dirs:
		allowed_dirs = ["Long"]
	overall_df = overall_df[overall_df["Direction"].isin(allowed_dirs)]
	if overall_df.empty:
		print("[Skip] No overall rows match the enabled trade directions.")
		return []
	if SYMBOLS:
		allowed_symbols = [sym.strip() for sym in SYMBOLS if sym and sym.strip()]
		if allowed_symbols:
			overall_df = overall_df[overall_df["Symbol"].isin(allowed_symbols)]
			if overall_df.empty:
				print(f"[Skip] Overall summary enthält keine Einträge für {', '.join(allowed_symbols)}.")
				return []
	group_cols = ["Indicator", "HTF"] if "HTF" in overall_df.columns else ["Indicator"]
	updated_all_rows = []
	aggregate_sections = []
	for group_key, rows in overall_df.groupby(group_cols):
		if isinstance(group_key, tuple):
			indicator_key = group_key[0]
			htf_value = group_key[1] if len(group_key) > 1 else HIGHER_TIMEFRAME
		else:
			indicator_key = group_key
			htf_value = HIGHER_TIMEFRAME
		if indicator_key not in INDICATOR_PRESETS:
			print(f"[Skip] Unknown indicator in overall file: {indicator_key}")
			continue
		apply_indicator_type(indicator_key)
		htf_str = str(htf_value) if not (isinstance(htf_value, float) and math.isnan(htf_value)) else HIGHER_TIMEFRAME
		htf_str = htf_str.strip()
		if not htf_str:
			htf_str = HIGHER_TIMEFRAME
		apply_higher_timeframe(htf_str)
		title = f"Overall-Beste Parameter ({INDICATOR_DISPLAY_NAME} {htf_str})"
		print(f"[Run] Indicator={INDICATOR_DISPLAY_NAME}, HTF={htf_str}, Kombinationen={len(rows)})")
		refreshed_rows = _run_saved_rows(rows, table_title=title, aggregate_sections=aggregate_sections)
		if refreshed_rows:
			updated_all_rows.extend(refreshed_rows)
	if updated_all_rows:
		updated_df = pd.DataFrame(updated_all_rows)
		os.makedirs(BASE_OUT_DIR, exist_ok=True)
		updated_df.to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")
	write_flat_trade_list(updated_all_rows)
	write_combined_overall_best_report(aggregate_sections)
	return updated_all_rows


def run_current_configuration():
	os.makedirs(OUT_DIR, exist_ok=True)
	if RUN_PARAMETER_SWEEP:
		return run_parameter_sweep()
	elif RUN_SAVED_PARAMS:
		run_saved_params()
	elif RUN_OVERALL_BEST:
		run_overall_best_params()
	else:
		print("[Skip] Backtesting disabled. Enable RUN_PARAMETER_SWEEP or RUN_SAVED_PARAMS.")
	return []


apply_indicator_type("supertrend")
apply_higher_timeframe(HIGHER_TIMEFRAME)


if __name__ == "__main__":
	if RUN_OVERALL_BEST:
		run_overall_best_params()
	else:
		indicator_candidates = get_indicator_candidates()
		htf_candidates = get_highertimeframe_candidates()
		if RUN_PARAMETER_SWEEP and CLEAR_BASE_OUTPUT_ON_SWEEP:
			clear_sweep_targets(indicator_candidates, htf_candidates)
		for indicator_name in indicator_candidates:
			apply_indicator_type(indicator_name)
			for htf_value in htf_candidates:
				apply_higher_timeframe(htf_value)
				print(f"[Run] Indicator={INDICATOR_DISPLAY_NAME}, HTF={HIGHER_TIMEFRAME}")
				summary_rows = run_current_configuration()
				record_global_best(indicator_name, summary_rows)
		write_overall_result_tables()
