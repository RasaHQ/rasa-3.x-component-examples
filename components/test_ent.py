import pytest
import pathlib

from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .nlu_ent import CapitalisedEntityExtractor


@pytest.fixture
def entity_extractor(tmpdir):
    """Generate a tfidf vectorizer with a tmpdir as the model storage."""
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)
    return CapitalisedEntityExtractor(
        config=CapitalisedEntityExtractor.get_default_config(),
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )


tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)


@pytest.mark.parametrize(
    "text, expected",
    [("hello World", ["World"]), ("Hello world", ["Hello"]), ("hello there world", [])],
)
def test_sparse_feats_added(entity_extractor, text, expected):
    """Checks if the sizes are appropriate."""
    # Create a message
    msg = Message({"text": text})

    # Process will process a list of Messages
    tokeniser.process([msg])
    entity_extractor.process([msg])
    # Check that the message has been processed correctly
    entities = msg.get("entities")
    assert [e["value"] for e in entities] == expected
