"""This file is used to test some of the basic
functionalities.
"""
from typing import Text, Dict
import inspect

def testing_function(
    a: Text,
    b: Dict[Text, Text]
):
    """A testing function to show that we
    are able to examine the function signature.
    """
    pass


inspect.signature(testing_function)