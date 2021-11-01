from rich import print 

import logging
from typing import Any, Text, Dict, List, Type
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
   DENSE_FEATURIZABLE_ATTRIBUTES,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
)
from joblib import dump, load
from rasa.shared.nlu.constants import TEXT
  
logger = logging.getLogger(__name__)
 

@DefaultV1Recipe.register(
   DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LogisticRegressionClassifier(IntentClassifier, GraphComponent):
 
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]
 
    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]
 
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
           "class_weight": "balanced",
           "max_iter": 100,
           "solver": "lbfgs"
        }
    
    def __init__(
            self, 
            config: Dict[Text, Any], 
            name: Text, 
            model_storage: ModelStorage,
            resource: Resource,
        ) -> None:
        self.name = name
        self.clf = LogisticRegression(solver=config["solver"], max_iter=config["max_iter"], class_weight=config["class_weight"])

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource
    
    def _create_training_matrix(self, training_data: TrainingData) -> None:
        X = []
        y = []
        for e in training_data.training_examples:
            if e.get(INTENT):
                # First element is sequence features, second is sentence features
                sparse_feats = e.get_sparse_features(TEXT)[1]
                # First element is sequence features, second is sentence features
                dense_feats = e.get_dense_features(TEXT)[1]
                if sparse_feats and dense_feats:
                    together = hstack([csr_matrix(sparse_feats.features), csr_matrix(dense_feats.features)])
                    X.append(together)
                    y.append(e.get(INTENT))
        return vstack(X), y

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
            self._set_intent(message)
       return messages

    def _set_intent(self, message: Message) -> None:
        pred = self.clf.predict([message.get(TEXT)])[0]
        probas = self.clf.predict_proba([message.get(TEXT)])[0]

        intent = {"name": pred, "confidence": probas[0]}
        intents = self.clf.classes_
        intent_ranking = {k: v for i, (k, v) in enumerate(zip(intents, probas)) if i < LABEL_RANKING_LENGTH}
        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
 
    def persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            dump(self.clf, model_dir / f"{self.name}.joblib")
    
    @classmethod
    def load(cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        with model_storage.read_from(resource) as model_dir:
            tfidfvectorizer = load(model_dir / f"{self.name}.joblib")
            component = cls(config, execution_context.node_name, model_storage, resource)
            component.clf = tfidfvectorizer
            return component

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data
   
    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
