import pytest

from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext

from .nlu_tok import AnotherWhitespaceTokenizer

node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
context = ExecutionContext(node_storage, node_resource)


tok_alphanum = AnotherWhitespaceTokenizer.create(
    config={
        "only_alphanum": True,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    },
    model_storage=node_storage,
    resource=node_resource,
    execution_context=context,
)

tok_no_alphanum = AnotherWhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)

tokenisers = [tok_alphanum, tok_no_alphanum]


@pytest.mark.parametrize("tok", tokenisers)
def test_base_use(tok):
    # Create a message
    msg = Message({"text": "hello world there"})

    # Process will process a list of Messages
    tok.process([msg])

    # Check that the message has been processed correctly
    assert [t.text for t in msg.get("text_tokens")] == ["hello", "world", "there"]


def test_specific_behavior():
    msg = Message({"text": "hello world 12345"})

    tok_no_alphanum.process([msg])
    assert [t.text for t in msg.get("text_tokens")] == ["hello", "world", "12345"]

    msg = Message({"text": "hello world #%!#$!#$"})

    # Process will process a list of Messages
    tok_alphanum.process([msg])
    assert [t.text for t in msg.get("text_tokens")] == [
        "hello",
        "world",
    ]
