from typing import NamedTuple, get_type_hints
class Test(NamedTuple):
    tmp: int
    tmp2: int
    
    
#def is_namedtuple_class(cls) -> bool:
#    return (
#        isinstance(cls, type)
#        and issubclass(cls, tuple)
#        and hasattr(cls, "_fields")
#        and hasattr(cls, "_asdict")
#        and isinstance(getattr(cls, "_fields"), tuple)
#    )
#
#print(is_namedtuple_class(object()))

