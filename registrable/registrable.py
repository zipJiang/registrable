"""Define a customized registrable class that
can be inherited to give customized behavior of registration.
"""
from typing import Text, Dict, List, Callable, TypeVar, Any, Tuple, Optional, Type
from typing import NewType, Union
from .lazy import Lazy
from copy import deepcopy
import inspect
import collections
from collections import defaultdict
from typing import Mapping


T = TypeVar("T")
Constructor = Callable[[Any], Tuple[T, Optional[Text]]]


_NO_DEFAULT = inspect.Parameter.empty


def infer_method_params(
    cls: Type[T],
    method: Optional[Callable] = None
) -> Dict[Text, inspect.Parameter]:
    if method is None:
        method = cls.__init__
    signature = inspect.signature(method)
    parameters = dict(signature.parameters)
    
    has_kwargs = False
    var_positional_key = None
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == param.VAR_POSITIONAL:
            var_positional_key = param.name
            
    if var_positional_key:
        del parameters[var_positional_key]
        
    if not has_kwargs:
        return parameters
    
    super_class = None
    for super_class_candidate in cls.mro()[1:]:
        if issubclass(super_class_candidate, Registrable):
            super_class = super_class_candidate
            break
        
    if super_class:
        super_parameters = infer_method_params(super_class)
    else:
        super_parameters
        
    return {**super_parameters, **parameters}


def takes_arg(obj, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise TypeError(f"object {obj} is not callable")
    return arg in signature.parameters


def takes_kwargs(obj) -> bool:
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise KeyError(f"object {obj} is not callable")
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD  # type: ignore
        for p in signature.parameters.values()
    )



def create_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dictionary of extra arguments, returns a dictionary of
    kwargs that actually are a part of the signature of the cls.from_params
    (or cls) method.
    """
    subextras: Dict[str, Any] = {}
    if hasattr(cls, "from_params"):
        from_params_method = cls.from_params  # type: ignore
    else:
        # In some rare cases, we get a registered subclass that does _not_ have a
        # from_params method (this happens with Activations, for instance, where we
        # register pytorch modules directly).  This is a bit of a hack to make those work,
        # instead of adding a `from_params` method for them somehow. Then the extras
        # in the class constructor are what we are looking for, to pass on.
        from_params_method = cls
    if takes_kwargs(from_params_method):
        # If annotation.params accepts **kwargs, we need to pass them all along.
        # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
        # object, but `TextFieldEmbedder.from_params` does not.
        subextras = extras
    else:
        # Otherwise, only supply the ones that are actual args; any additional ones
        # will cause a TypeError.
        subextras = {k: v for k, v in extras.items() if takes_arg(from_params_method, k)}
    return subextras


def can_construct_from_params(type_: Type) -> bool:
    if type_ in [str, int, float, bool]:
        return True
    if origin == Lazy:
        return True
    elif origin:
        origin = getattr(type_, "__origin__", None)
        if hasattr(type_, "from_params"):
            return True
        args = getattr(type_, "__args__")
        return all(can_construct_from_params(arg) for arg in args)

    return hasattr(type_, "from_params")


def construct_arg(
    class_name: Text,
    argument_name: Text,
    popped_params: Dict[Text, Any],
    annotation: Type,
    default: Any,
    **extras
) -> Any:
    
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', None)
    
    optional = default != _NO_DEFAULT
    
    if hasattr(annotation, 'from_params'):
        if popped_params is default:
            return default
        elif popped_params is not None:
            subextras = create_extras(annotation, extras)
            
            if isinstance(popped_params, str):
                popped_params = {"type": popped_params}
                
            result = annotation.from_params(params=popped_params, **subextras)
            
            return result
        
        elif not optional:
            raise ValueError(f"Argument {argument_name} of {class_name} is not optional.")
        
        else:
            return default
        
    elif annotation in {int, bool}:
        if type(popped_params) in {int, bool, str}:
            return annotation(popped_params)
        else:
            raise TypeError(f"Argument {argument_name} of {class_name} is not of type {annotation}.")
        
    elif annotation == float:
        if type(popped_params) in {int, float}:
            return popped_params
        else:
            raise TypeError(f"Argument {argument_name} of {class_name} is not of type {annotation}.")
        
    elif origin == Lazy:
        if popped_params is default:
            return default
        
        value_cls = args[0]
        subextras = create_extras(value_cls, extras)
        return Lazy(value_cls, params=deepcopy(popped_params), constructor_extras=subextras)
        
    elif origin == Union:
        backup_params = deepcopy(popped_params)
        
        error_chain: Optional[Exception] = None
        for arg_annotation in args:
            try:
                return construct_arg(
                    str(arg_annotation),
                    argument_name,
                    popped_params,
                    arg_annotation,
                    default,
                    **extras
                )
            except (TypeError, ValueError, AttributeError) as e:
                popped_params = deepcopy(backup_params)
                e.args = (f"While constructing an argument of type {arg_annotation}",) + e.args
                e.__cause__ = error_chain
                error_chain = e
                
    else:
        return popped_params
        

def pop_and_construct_arg(
    class_name: Text,
    argument_name: Text,
    annotation: Type,
    default: Any,
    params: Dict[Text, Any],
    **extras
) -> Any:
    """Construct an constructable argument from the params
    """
    
    if argument_name in extras:
        if argument_name not in params:
            return extras[argument_name]
    
    popped_arg = params.pop(argument_name, default) if default != _NO_DEFAULT else params.pop(argument_name)
    
    if popped_arg is None:
        return None
    
    return construct_arg(class_name, argument_name, popped_arg, annotation, default, **extras)
    
    
def create_kwargs(
    constructor: Callable[..., T],
    cls: Type[T],
    params: Dict[Text, Any],
    **extras
) -> Dict[Text, Any]:
    """
    """
    kwargs: Dict[str, Any] = {}
    parameters = infer_method_params(cls, constructor)
    accepts_kwargs = False
    
    for param_name, param in parameters.items():
        if param_name == 'self':
            continue
        
        if param.kind == param.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        
        annotation = param.annotation
        explicitly_set  = param_name in params
        
        constructed_arg = pop_and_construct_arg(
            cls.__name__,
            param_name,
            annotation,
            param.default,
            params,
            **extras
        )
        
        if explicitly_set or constructed_arg is not param.default:
            kwargs[param_name] = constructed_arg
    
    if accepts_kwargs:
        kwargs.update(params)
        
    else:
        assert params == {}, f"Extra parameters passed to {cls.__name__}: {params}"
        
    return kwargs

class NameAlreadyRegisteredError(Exception):
    def __init__(self, new_name: Text, registration_table: Dict[Text, Callable[[Any], T]]):
        self._new_name = new_name
        super().__init__(f"{new_name} is already registered for callable {registration_table[new_name]}.")


class Registrable:
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
    def from_params(cls, params) -> T:
        """This function is called to generate a class obj from
        a parameter dictionary that matches the callable.
        """
        class_name = params.pop('type')
        class_tuple = cls.__named_subclasses__[cls][class_name]

        class_, constructor = class_tuple

        # need to recursively call from_params with the kwargs
        # TODO: make sure it does not look for object.__init__
        kwargs = create_kwargs(constructor, class_, params)

        if constructor is None:
            return class_(**kwargs)
        
        else:
            return getattr(class_, constructor)(**kwargs)
