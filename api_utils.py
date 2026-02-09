"""API utilities for improved robustness and error handling.

This module provides:
- Circuit Breaker pattern to prevent cascade failures
- API key validation with actual connection testing
- Descriptive error messages with solutions
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, TypeVar, Any

import requests


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5      # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2      # Successes in half-open before closing


class CircuitBreaker:
    """Circuit breaker to prevent cascade failures during API outages.

    Usage:
        breaker = CircuitBreaker(name="binance-api")

        try:
            result = breaker.call(lambda: api_request())
        except CircuitOpenError:
            # API is down, use cached data or skip
            pass
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    print(f"[CircuitBreaker:{self.name}] HALF_OPEN - testing recovery")
            return self._state

    def call(self, func: Callable[[], Any]) -> Any:
        """Execute function through circuit breaker.

        Raises:
            CircuitOpenError: If circuit is open (API unavailable)
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN - API temporarily unavailable. "
                f"Will retry in {self._time_until_recovery():.0f}s"
            )

        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    print(f"[CircuitBreaker:{self.name}] CLOSED - service recovered")
            else:
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                print(f"[CircuitBreaker:{self.name}] OPEN - recovery failed")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                print(f"[CircuitBreaker:{self.name}] OPEN - {self._failure_count} failures")

    def _time_until_recovery(self) -> float:
        """Get seconds until circuit will try half-open."""
        if self._last_failure_time is None:
            return 0
        elapsed = time.time() - self._last_failure_time
        return max(0, self.config.recovery_timeout - elapsed)

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ============================================================================
# Global Circuit Breakers
# ============================================================================

# Shared circuit breakers for different API endpoints
_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name)
        return _circuit_breakers[name]


# Pre-defined circuit breakers
BINANCE_SPOT_CB = "binance-spot"
BINANCE_FUTURES_CB = "binance-futures"


# ============================================================================
# API Key Validation
# ============================================================================

@dataclass
class ApiKeyValidationResult:
    """Result of API key validation."""
    valid: bool
    error_message: Optional[str] = None
    solution: Optional[str] = None
    permissions: Optional[list[str]] = None


def validate_binance_api_keys(
    api_key: str,
    api_secret: str,
    testnet: bool = False
) -> ApiKeyValidationResult:
    """Validate Binance API keys by testing actual API connection.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: If True, validate against testnet endpoints

    Returns:
        ApiKeyValidationResult with validation status and details
    """
    import hmac
    import hashlib

    if not api_key or not api_key.strip():
        return ApiKeyValidationResult(
            valid=False,
            error_message="API Key fehlt",
            solution="Setze BINANCE_API_KEY in der .env Datei oder als Umgebungsvariable"
        )

    if not api_secret or not api_secret.strip():
        return ApiKeyValidationResult(
            valid=False,
            error_message="API Secret fehlt",
            solution="Setze BINANCE_API_SECRET in der .env Datei oder als Umgebungsvariable"
        )

    # Choose endpoint based on testnet flag
    if testnet:
        base_url = "https://testnet.binancefuture.com"
        endpoint = "/fapi/v1/account"
    else:
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/account"

    # Create signed request
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        api_secret.encode(),
        query_string.encode(),
        hashlib.sha256
    ).hexdigest()

    url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            permissions = data.get("permissions", [])
            return ApiKeyValidationResult(
                valid=True,
                permissions=permissions
            )

        # Handle specific error codes
        error_code = None
        error_msg = ""
        try:
            error_data = response.json()
            error_code = error_data.get("code")
            error_msg = error_data.get("msg", "")
        except:
            error_msg = response.text

        error_solutions = {
            -2015: ("Ungueltige API Key oder Signatur",
                   "Pruefe ob API Key und Secret korrekt kopiert wurden (keine Leerzeichen)"),
            -1022: ("Signatur ungueltig",
                   "API Secret ist falsch oder wurde geaendert"),
            -2014: ("API Key nicht aktiviert",
                   "Aktiviere den API Key in den Binance-Einstellungen"),
            -1021: ("Timestamp ausserhalb des Fensters",
                   "Synchronisiere die Systemzeit (ntpdate oder timesyncd)"),
        }

        if error_code in error_solutions:
            msg, solution = error_solutions[error_code]
            return ApiKeyValidationResult(
                valid=False,
                error_message=f"{msg} (Code: {error_code})",
                solution=solution
            )

        return ApiKeyValidationResult(
            valid=False,
            error_message=f"API-Fehler: {error_msg} (Code: {error_code})",
            solution="Pruefe die Binance API-Dokumentation fuer diesen Fehlercode"
        )

    except requests.exceptions.Timeout:
        return ApiKeyValidationResult(
            valid=False,
            error_message="Verbindungs-Timeout",
            solution="Pruefe die Internetverbindung oder versuche es spaeter erneut"
        )
    except requests.exceptions.ConnectionError:
        return ApiKeyValidationResult(
            valid=False,
            error_message="Keine Verbindung zur Binance API",
            solution="Pruefe Internetverbindung und Firewall-Einstellungen"
        )
    except Exception as e:
        return ApiKeyValidationResult(
            valid=False,
            error_message=f"Unerwarteter Fehler: {e}",
            solution="Pruefe die Log-Ausgabe fuer Details"
        )


def validate_all_api_keys() -> dict[str, ApiKeyValidationResult]:
    """Validate all configured API keys (spot + testnet).

    Returns:
        Dictionary with validation results for each key type
    """
    results = {}

    # Spot/Live API keys
    spot_key = os.getenv("BINANCE_API_KEY", "")
    spot_secret = os.getenv("BINANCE_API_SECRET", "")
    if spot_key or spot_secret:
        print("[API-Check] Pruefe Spot/Live API Keys...")
        results["spot"] = validate_binance_api_keys(spot_key, spot_secret, testnet=False)
        if results["spot"].valid:
            print("[API-Check] Spot API Keys: OK")
        else:
            print(f"[API-Check] Spot API Keys: FEHLER - {results['spot'].error_message}")
            if results["spot"].solution:
                print(f"            Loesung: {results['spot'].solution}")

    # Testnet API keys
    test_key = os.getenv("BINANCE_API_KEY_TEST", "")
    test_secret = os.getenv("BINANCE_API_SECRET_TEST", "")
    if test_key or test_secret:
        print("[API-Check] Pruefe Testnet API Keys...")
        results["testnet"] = validate_binance_api_keys(test_key, test_secret, testnet=True)
        if results["testnet"].valid:
            print("[API-Check] Testnet API Keys: OK")
        else:
            print(f"[API-Check] Testnet API Keys: FEHLER - {results['testnet'].error_message}")
            if results["testnet"].solution:
                print(f"            Loesung: {results['testnet'].solution}")

    return results


# ============================================================================
# Enhanced Error Messages
# ============================================================================

API_ERROR_SOLUTIONS = {
    # Rate limiting
    429: {
        "message": "Rate-Limit erreicht",
        "solution": "Reduziere die Anfragenhaeufigkeit oder warte kurz",
        "action": "retry_with_backoff"
    },
    418: {
        "message": "IP wurde temporaer gesperrt (zu viele Anfragen)",
        "solution": "Warte 5-30 Minuten oder nutze eine andere IP",
        "action": "wait_long"
    },

    # Authentication
    401: {
        "message": "Authentifizierung fehlgeschlagen",
        "solution": "Pruefe API Key und Secret in der .env Datei",
        "action": "check_credentials"
    },
    403: {
        "message": "Zugriff verweigert",
        "solution": "API Key hat keine Berechtigung fuer diese Operation",
        "action": "check_permissions"
    },

    # Server errors
    500: {
        "message": "Interner Server-Fehler bei Binance",
        "solution": "Versuche es in ein paar Sekunden erneut",
        "action": "retry_with_backoff"
    },
    502: {
        "message": "Bad Gateway - Binance Server-Problem",
        "solution": "Versuche es in ein paar Sekunden erneut",
        "action": "retry_with_backoff"
    },
    503: {
        "message": "Binance API voruebergehend nicht verfuegbar",
        "solution": "Warte auf Wiederherstellung des Service",
        "action": "wait_medium"
    },

    # Client errors
    400: {
        "message": "Ungueltige Anfrage",
        "solution": "Pruefe die Anfrage-Parameter",
        "action": "log_details"
    },
    404: {
        "message": "Ressource nicht gefunden",
        "solution": "Pruefe Symbol-Name oder Endpoint",
        "action": "check_symbol"
    },
}


def get_error_solution(status_code: int, error_text: str = "") -> tuple[str, str]:
    """Get user-friendly error message and solution for HTTP status code.

    Returns:
        Tuple of (error_message, solution)
    """
    if status_code in API_ERROR_SOLUTIONS:
        info = API_ERROR_SOLUTIONS[status_code]
        return info["message"], info["solution"]

    if 400 <= status_code < 500:
        return f"Client-Fehler ({status_code})", "Pruefe die Anfrage-Parameter"

    if 500 <= status_code < 600:
        return f"Server-Fehler ({status_code})", "Versuche es spaeter erneut"

    return f"HTTP-Fehler {status_code}", error_text or "Unbekannter Fehler"


# ============================================================================
# Startup Health Check
# ============================================================================

def run_startup_health_check(skip_api_check: bool = False) -> bool:
    """Run health checks at application startup.

    Args:
        skip_api_check: If True, skip actual API validation (for offline testing)

    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 60)
    print("[Health-Check] Starte Systemueberpruefung...")
    print("=" * 60)

    all_ok = True

    # Check 1: Environment variables
    print("\n[1/3] Pruefe Umgebungsvariablen...")
    required_vars = ["BINANCE_API_KEY_TEST", "BINANCE_API_SECRET_TEST"]
    optional_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]

    missing_required = [v for v in required_vars if not os.getenv(v)]
    missing_optional = [v for v in optional_vars if not os.getenv(v)]

    if missing_required:
        print(f"      WARNUNG: Fehlende Variablen: {', '.join(missing_required)}")
        all_ok = False
    else:
        print("      Testnet-Credentials: Vorhanden")

    if missing_optional:
        print(f"      Info: Live-Credentials nicht gesetzt: {', '.join(missing_optional)}")
    else:
        print("      Live-Credentials: Vorhanden")

    # Check 2: API connectivity (optional)
    if not skip_api_check:
        print("\n[2/3] Teste API-Verbindung...")
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=5
            )
            if response.status_code == 200:
                print("      Binance API: Erreichbar")
            else:
                print(f"      Binance API: Fehler ({response.status_code})")
                all_ok = False
        except Exception as e:
            print(f"      Binance API: Nicht erreichbar ({e})")
            all_ok = False

        # Validate API keys
        print("\n[3/3] Validiere API Keys...")
        results = validate_all_api_keys()
        for key_type, result in results.items():
            if not result.valid:
                all_ok = False
    else:
        print("\n[2/3] API-Verbindungstest: Uebersprungen")
        print("\n[3/3] API-Key-Validierung: Uebersprungen")

    print("\n" + "=" * 60)
    if all_ok:
        print("[Health-Check] Alle Pruefungen bestanden")
    else:
        print("[Health-Check] Einige Pruefungen fehlgeschlagen - siehe oben")
    print("=" * 60 + "\n")

    return all_ok
