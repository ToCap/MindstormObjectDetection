"""
Description and access of the adjustable parameters of a detection module.

A module keeps its parameters as global variables (upper case, the unit in
the name) and lists the ones that can be changed at run time in a table of
`Parameter`. It exposes them with get_parameters / get_parameter /
set_parameter, which call the functions below.
"""
import numbers
from collections import namedtuple

# type: int or float; minimum / maximum: inclusive bounds;
# unit: text ("px", "m", "ratio", ...); odd: the value must be odd
Parameter = namedtuple(
    "Parameter",
    "name type minimum maximum unit description odd",
    defaults=(False,),
)


def _error(tag, code_name, code, message):
    return ValueError("[%s] %s (0x%02X): %s" % (tag, code_name, code, message))


def _find(table, name, tag, code_name, code):
    for parameter in table:
        if parameter.name == name:
            return parameter
    raise _error(tag, code_name, code, "unknown parameter '%s'" % name)


def describe(namespace, table):
    """List of dict (name, type, minimum, maximum, unit, description, value)."""
    return [
        {
            "name": p.name,
            "type": p.type.__name__,
            "minimum": p.minimum,
            "maximum": p.maximum,
            "unit": p.unit,
            "description": p.description,
            "value": namespace[p.name],
        }
        for p in table
    ]


def get_value(namespace, table, name, tag, code_name, code):
    _find(table, name, tag, code_name, code)
    return namespace[name]


def set_value(namespace, table, name, value, tag, code_name, code):
    """Checks the value and stores it in the namespace (the module globals)."""
    parameter = _find(table, name, tag, code_name, code)

    if isinstance(value, bool):
        raise _error(tag, code_name, code, "'%s': a boolean is not a number" % name)
    if parameter.type is int:
        if not isinstance(value, numbers.Integral):
            raise _error(tag, code_name, code, "'%s': an integer is expected" % name)
        value = int(value)
    else:
        if not isinstance(value, numbers.Real):
            raise _error(tag, code_name, code, "'%s': a number is expected" % name)
        value = float(value)

    if not parameter.minimum <= value <= parameter.maximum:
        raise _error(
            tag, code_name, code,
            "'%s': %s is out of range [%s, %s]"
            % (name, value, parameter.minimum, parameter.maximum)
        )
    if parameter.odd and value % 2 == 0:
        raise _error(tag, code_name, code, "'%s': an odd value is expected" % name)

    namespace[name] = value
