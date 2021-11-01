import logging
from typing import Any, Text, Dict, List, Type
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    TOKENS,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from rasa.shared.nlu.constants import TEXT
  
logger = logging.getLogger(__name__)
 

@DefaultV1Recipe.register(
   DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class CapitalisedEntityExtractor(IntentClassifier, GraphComponent):
 
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Tokenizer]
 
    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []
 
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"name": "THING"}
    
    def __init__(
            self, 
            config: Dict[Text, Any], 
            name: Text, 
            model_storage: ModelStorage,
            resource: Resource,
        ) -> None:
        self.name = name
        self.entity_name = config.get("name")
        
        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource
    
    def train(
        self, training_data: TrainingData
    ) -> Resource:

        X, y = self._create_training_matrix(training_data)

        self.clf.fit(X, y)
        self.persist()

        return self._resource
 
    @classmethod
    def create(
       cls,
       config: Dict[Text, Any],
       model_storage: ModelStorage,
       resource: Resource,
       execution_context: ExecutionContext,
    ) -> GraphComponent:
       return cls(config, execution_context.node_name, model_storage, resource)
 
    def process(self, messages: List[Message]) -> List[Message]:
       for message in messages:
            self._set_entities(message)
       return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        tokens: List[Token] = message.get(TOKENS)
        extracted_entities = [] 
        for token in tokens:
            if token.text[0].isupper():
                extracted_entities.append({
                    ENTITY_ATTRIBUTE_TYPE: self.entity_name,
                    ENTITY_ATTRIBUTE_START: token.start,
                    ENTITY_ATTRIBUTE_END: token.end,
                    ENTITY_ATTRIBUTE_VALUE: token.text,
                    "confidence": 1.0,
                })
        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data
   
    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
