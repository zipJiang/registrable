"""This file is used to test some of the basic
functionalities.
"""
from typing import Text, Dict, List
from registrable.registrable import Registrable
from registrable.lazy import Lazy


class ModelClass(Registrable):
    def __init__(
        self,
        name: Text,
    ):
        self.name = name
        

@ModelClass.register("vocab_model")
class VocabularyModelClass(ModelClass):
    def __init__(
        self,
        name: Text,
        vocab_size: int,
    ):
        super().__init__(name)
        self.vocab_size = vocab_size


class Trainer(Registrable):
    """This is a trainer class."""

    def __init__(
        self,
        name: Text,
    ):
        """Initialize the trainer class."""
        self.name = name
        
        
@Trainer.register("custom_trainer")
class CustomTrainer(Trainer):
    def __init__(
        self,
        name: Text,
        models: List[Lazy[ModelClass]]
    ):
        """
        """
        super().__init__(name)
        self.models = [model.construct(vocab_size=100) for model in models]
        

def test_register():
    params = {
        "type": "custom_trainer",
        "name": "my_trainer",
        "models": [
            {
                "type": "vocab_model",
                "name": "my_model",
            },
            {
                "type": "vocab_model",
                "name": "my_model2",
            }
        ],
    }
    
    trainer = Trainer.from_params(params)
    
    print(trainer.models[1].vocab_size)

    
test_register()