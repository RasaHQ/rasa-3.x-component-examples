import pytest

from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .nlu_dense import BytePairFeaturizer

node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
context = ExecutionContext(node_storage, node_resource)


tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)
bpemb_feat = BytePairFeaturizer(
    config={
        "lang": "en",
        "dim": 25,
        "vs": 1000,
        "alias": "foobar",
        "vs_fallback": True,
    },
    name=context.node_name,
)


@pytest.mark.parametrize(
    "text, expected", [("hello", 1), ("hello world", 2), ("hello there world", 3)]
)
def test_dense_feats_added(text, expected):
    """Checks if the sizes are appropriate."""
    # Create a message
    msg = Message({"text": text})

    # Process will process a list of Messages
    tokeniser.process([msg])
    bpemb_feat.process([msg])

    # Check that the message has been processed correctly
    seq_feats, sent_feats = msg.get_dense_features("text")
    assert seq_feats.features.shape == (expected, 25)
    assert sent_feats.features.shape == (1, 25)
