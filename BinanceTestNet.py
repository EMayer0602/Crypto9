import os
import time
import hmac
import hashlib
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from dotenv import load_dotenv

# .env laden (aus aktuellem Verzeichnis)
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY_TEST")
API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")

BASE_URL = "https://testnet.binance.vision"

# Konfiguration
REQUEST_TIMEOUT = 30  # Sekunden
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2s, 4s, 8s


def validate_api_keys():
    """Validiert, dass API-Keys konfiguriert sind."""
    if not API_KEY or not API_KEY.strip():
        raise ValueError("BINANCE_API_KEY_TEST ist nicht konfiguriert in .env")
    if not API_SECRET or not API_SECRET.strip():
        raise ValueError("BINANCE_API_SECRET_TEST ist nicht konfiguriert in .env")
    if len(API_KEY) < 20:
        raise ValueError("BINANCE_API_KEY_TEST scheint ungültig zu sein (zu kurz)")
    if len(API_SECRET) < 20:
        raise ValueError("BINANCE_API_SECRET_TEST scheint ungültig zu sein (zu kurz)")


def sign_request(params, secret):
    """Signiert die Request-Parameter mit HMAC-SHA256."""
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature


def _retry_request(request_func, max_retries=MAX_RETRIES):
    """
    Führt einen Request mit exponential backoff retry aus.

    Args:
        request_func: Eine Funktion die den Request ausführt und Response zurückgibt
        max_retries: Maximale Anzahl an Wiederholungsversuchen

    Returns:
        Response-Objekt bei Erfolg

    Raises:
        RequestException: Nach allen fehlgeschlagenen Versuchen
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = request_func()

            # Bei Rate-Limit (429) oder Server-Fehler (5xx): Retry
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RETRY_BACKOFF_BASE ** attempt))
                print(f"Rate-Limit erreicht. Warte {retry_after}s (Versuch {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_after)
                continue
            elif response.status_code >= 500:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                print(f"Server-Fehler {response.status_code}. Warte {wait_time}s (Versuch {attempt + 1}/{max_retries + 1})")
                time.sleep(wait_time)
                continue

            return response

        except (Timeout, ConnectionError) as e:
            last_exception = e
            wait_time = RETRY_BACKOFF_BASE ** attempt
            print(f"Netzwerkfehler: {e}. Warte {wait_time}s (Versuch {attempt + 1}/{max_retries + 1})")
            time.sleep(wait_time)
            continue

    raise RequestException(f"Request fehlgeschlagen nach {max_retries + 1} Versuchen: {last_exception}")


def place_order(symbol, side, order_type, quantity, price=None):
    """
    Platziert eine Order auf Binance Testnet.

    Args:
        symbol: Trading-Paar (z.B. "BTCUSDT")
        side: "BUY" oder "SELL"
        order_type: "LIMIT" oder "MARKET"
        quantity: Menge
        price: Preis (nur für LIMIT orders)

    Returns:
        dict: API-Response bei Erfolg

    Raises:
        ValueError: Bei ungültigen API-Keys
        RequestException: Bei Netzwerkfehlern nach Retries
    """
    # API-Keys validieren
    validate_api_keys()

    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "side": side,              # "BUY" oder "SELL"
        "type": order_type,        # "LIMIT" oder "MARKET"
        "timeInForce": "GTC",      # nur für LIMIT nötig
        "quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }

    if price:
        params["price"] = price

    query = sign_request(params, API_SECRET)

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    def do_request():
        return requests.post(url, headers=headers, data=query, timeout=REQUEST_TIMEOUT)

    response = _retry_request(do_request)

    # Status-Code prüfen
    if response.status_code != 200:
        error_msg = f"API-Fehler {response.status_code}"
        try:
            error_data = response.json()
            error_msg += f": {error_data.get('msg', response.text)}"
        except ValueError:
            error_msg += f": {response.text}"
        raise RequestException(error_msg)

    # Response parsen
    try:
        return response.json()
    except ValueError as e:
        raise RequestException(f"Ungültige JSON-Response: {e}")

# Beispiel: Kaufe 0.001 BTC gegen USDT zu 90.000 USDT
order = place_order("BTCUSDT", "BUY", "LIMIT", 0.001, 90000)
print(order)
