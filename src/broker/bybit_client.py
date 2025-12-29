"""
src/broker/bybit_client.py

Cliente simple para consumir velas (kline) desde Bybit V5 (endpoint público).
Incluye:
- Manejo de rate limit (retCode 10006) con backoff + jitter
- Descarga robusta por paginación HACIA ATRÁS usando 'end' (más estable en Bybit V5)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import time
import random

import requests


@dataclass(frozen=True)
class KlineRow:
    start_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class BybitClient:
    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        timeout: int = 20,
        max_retries: int = 10,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        jitter_s: float = 0.35,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.jitter_s = jitter_s

    @staticmethod
    def dt_to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _parse_kline_list(raw_list: List[List[str]]) -> List[KlineRow]:
        rows: List[KlineRow] = []
        for item in raw_list:
            rows.append(
                KlineRow(
                    start_time_ms=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6]),
                )
            )
        return rows

    def _request_json_with_backoff(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                ret_code = int(data.get("retCode", -1))

                # Rate limit
                if ret_code == 10006:
                    wait = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
                    wait += random.uniform(0, self.jitter_s)
                    time.sleep(wait)
                    continue

                # Other API error
                if ret_code != 0:
                    raise RuntimeError(f"Bybit retCode != 0: {data}")

                return data

            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                wait = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
                wait += random.uniform(0, self.jitter_s)
                time.sleep(wait)

        if last_error:
            raise last_error
        raise RuntimeError("Fallo desconocido en request.")

    def get_kline_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        category: str = "linear",
        limit: int = 1000,
    ) -> List[KlineRow]:
        """
        Pide un batch de klines dentro de [start_ms, end_ms].
        Devuelve lista (puede venir en orden descendente según Bybit).
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
        }

        data = self._request_json_with_backoff(url, params)

        result = data.get("result", {})
        raw_list = result.get("list", [])
        if not raw_list:
            return []

        rows = self._parse_kline_list(raw_list)
        return rows

    def download_klines_backward(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        category: str = "linear",
        limit: int = 1000,
        sleep_s: float = 0.35,
        progress_every: int = 10,
    ) -> List[KlineRow]:
        """
        Descarga robusta paginando HACIA ATRÁS:
        - Empieza desde end_dt, pide un batch
        - Luego mueve end al timestamp más antiguo del batch - 1ms
        - Repite hasta llegar a start_dt

        Esto suele ser más estable con Bybit V5.
        """
        start_ms = self.dt_to_ms(start_dt)
        end_ms = self.dt_to_ms(end_dt)

        all_rows: Dict[int, KlineRow] = {}
        current_end = end_ms
        loops = 0

        while current_end > start_ms:
            batch = self.get_kline_batch(
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,        # start fijo
                end_ms=current_end,       # end va bajando
                category=category,
                limit=limit,
            )

            if not batch:
                break

            # Bybit suele devolver descendente; buscamos min/max
            oldest = min(batch, key=lambda r: r.start_time_ms)
            newest = max(batch, key=lambda r: r.start_time_ms)

            # Guardar en dict para dedupe
            for r in batch:
                all_rows[r.start_time_ms] = r

            # Mover ventana hacia atrás
            next_end = oldest.start_time_ms - 1
            if next_end >= current_end:
                # Protección anti-loop infinito
                break
            current_end = next_end

            loops += 1
            if progress_every > 0 and loops % progress_every == 0:
                # progreso ligero (no satura)
                print(f"[PROGRESS] batches={loops} | rows={len(all_rows)} | window_end={datetime.fromtimestamp(current_end/1000, tz=timezone.utc)}")

            time.sleep(sleep_s)

        rows = list(all_rows.values())
        rows.sort(key=lambda r: r.start_time_ms)
        return rows
