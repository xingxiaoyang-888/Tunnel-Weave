"""Shared class order and validation errors."""
CLASSES = ["crack", "surface_wetness", "spalling", "surface_deposit", "rust",
           "exposed_rebar", "fracture", "missing", "displacement", "inverted",
           "deformed", "detached", "foreign_object"]

class ContractError(ValueError):
    """Input does not satisfy the model or annotation contract."""

def require(value, message):
    if not value:
        raise ContractError(message)
