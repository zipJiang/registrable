"""Define a customized registrable class that
can be inherited to give customized behavior of registration.
"""
from typing import Text, Dict, List, Callable, TypeVar, Any, Tuple, Optional, Type
from typing import NewType
from collections import defaultdict


T = TypeVar("T")
Constructor = Callable[[Any], Tuple[T, Optional[Text]]]


class NameAlreadyRegisteredError(Exception):
    def __init__(self, new_name: Text, registration_table: Dict[Text, Callable[[Any], T]]):
        self._new_name = new_name
        super().__init__(f"{new_name} is already registered for callable {registration_table[new_name]}.")


class MyRegistrable:
    __named_subclasses__: Dict[Text, Tuple[Text, Tuple[Type[T], Text]]] = defaultdict(dict)
    
    def __init__(self):
        """Do nothing in the init function
        """
        pass
        
    @classmethod
    def register(cls, name: Text, constructor: Optional[Text] = None) -> Callable[[Type[T]], Type[T]]:
        """Register a class to the __named_subclasses dict of the group.
        """
        registry = cls.__named_subclasses__[cls]
        
        if name in registry:
            raise NameAlreadyRegisteredError(name, registry)
        
        def register_class(subclass: Type[T]) -> Type[T]:
            registry[name] = (subclass, constructor)
            
            return subclass
        
        return register_class

    @classmethod
    def from_params(cls, **kwargs) -> T:
        """This function is called to generate a class obj from
        a parameter dictionary that matches the callable.
        """
        class_name = kwargs.pop('type')
        class_tuple = cls.__named_subclasses__[cls][class_name]

        class_, constructor = class_tuple

        if constructor is None:
            return class_(**kwargs)
        
        else:
            return getattr(class_, constructor)(**kwargs)