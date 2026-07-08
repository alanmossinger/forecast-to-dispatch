"""Forecast-to-Dispatch: governed price-spread forecasting and battery dispatch.

Why this matters: grid-scale battery revenue comes from price spreads and rare
scarcity events, not average prices. This package forecasts the *distribution*
of wholesale power prices, co-optimizes a battery's energy + ancillary-service
dispatch against that forecast, and wraps the whole decision loop in an
audit-grade governance layer (NIST AI RMF / EU AI Act aligned) so the agent's
revenue claims are defensible to an operator, an ISO, or a risk committee.
"""

__version__ = "0.1.0"
