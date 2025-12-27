# IA_trade (ETH-first)

Proyecto de trading automatizado enfocado en **un solo par: ETH/USDT**.

## Objetivo del flujo
1. Entrenar offline con históricos (ETH).
2. Validar el modelo.
3. Operar en vivo (ETH).
4. Reentrenar periódicamente con resultados reales (batch/online update).
5. Dashboard basado únicamente en operaciones reales cerradas.

## Estructura
- `config/`: Configuración central (símbolo, timeframes, paths, risk).
- `data/`: Históricos y datasets (ignorado por git por defecto).
- `models/`: Modelos entrenados y checkpoints (ignorado por git por defecto).
- `logs/`: Logs y trades reales (ignorado por git por defecto, excepto README/.gitkeep).
- `src/`: Código del sistema.
  - `src/training/`: entrenamiento y evaluación
  - `src/live/`: ejecución en vivo y manejo de posiciones
  - `src/dashboard/`: dashboard de monitoreo
  - `src/env/`: entorno RL
  - `src/features/`: features e indicadores
  - `src/broker/`: cliente Bybit y helpers

## Scripts principales
- `src/training/train_offline.py`
- `src/training/evaluate.py`
- `src/live/live_trading.py`
- `src/training/online_update.py`
- `src/dashboard/dashboard.py`

## Notas
- Este repo no incluye claves API. Usar `.env` (ignorado por git).
- Los logs y modelos no se suben al repo por defecto.
