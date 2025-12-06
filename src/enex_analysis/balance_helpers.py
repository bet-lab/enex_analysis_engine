"""
Helper functions for creating energy, entropy, and exergy balance dictionaries.

This module provides utility functions to create structured balance dictionaries
for subsystems, reducing code duplication and improving maintainability.
"""


def create_energy_balance(subsystems):
    """
    Create an energy balance dictionary for given subsystems.
    
    Parameters:
    -----------
    subsystems : dict
        Dictionary mapping subsystem names to their balance entries.
        Structure: {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...}
            },
            ...
        }
    
    Returns:
    --------
    dict
        Energy balance dictionary with structure:
        {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...}
            },
            ...
        }
    
    Example:
    --------
    >>> subsystems = {
    ...     "hot water tank": {
    ...         "in": {"E_heater": 1000.0, "Q_w_sup_tank": 500.0},
    ...         "out": {"Q_w_tank": 1200.0, "Q_l_tank": 300.0}
    ...     }
    ... }
    >>> balance = create_energy_balance(subsystems)
    """
    return subsystems.copy()


def create_entropy_balance(subsystems):
    """
    Create an entropy balance dictionary for given subsystems.
    
    Parameters:
    -----------
    subsystems : dict
        Dictionary mapping subsystem names to their balance entries.
        Structure: {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...},
                "gen": {symbol: value, ...}
            },
            ...
        }
    
    Returns:
    --------
    dict
        Entropy balance dictionary with structure:
        {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...},
                "gen": {symbol: value, ...}
            },
            ...
        }
    
    Example:
    --------
    >>> subsystems = {
    ...     "hot water tank": {
    ...         "in": {"S_heater": 0.0, "S_w_sup_tank": 1.5},
    ...         "out": {"S_w_tank": 1.8, "S_l_tank": 0.2},
    ...         "gen": {"S_g_tank": 0.5}
    ...     }
    ... }
    >>> balance = create_entropy_balance(subsystems)
    """
    return subsystems.copy()


def create_exergy_balance(subsystems):
    """
    Create an exergy balance dictionary for given subsystems.
    
    Parameters:
    -----------
    subsystems : dict
        Dictionary mapping subsystem names to their balance entries.
        Structure: {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...},
                "con": {symbol: value, ...}
            },
            ...
        }
    
    Returns:
    --------
    dict
        Exergy balance dictionary with structure:
        {
            "subsystem_name": {
                "in": {symbol: value, ...},
                "out": {symbol: value, ...},
                "con": {symbol: value, ...}
            },
            ...
        }
    
    Example:
    --------
    >>> subsystems = {
    ...     "hot water tank": {
    ...         "in": {"X_heater": 1000.0, "X_w_sup_tank": 100.0},
    ...         "out": {"X_w_tank": 900.0, "X_l_tank": 50.0},
    ...         "con": {"X_c_tank": 150.0}
    ...     }
    ... }
    >>> balance = create_exergy_balance(subsystems)
    """
    return subsystems.copy()

