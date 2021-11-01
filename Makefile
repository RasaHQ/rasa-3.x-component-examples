flake:
	flake8 components --count --statistics --max-complexity=10 --max-line-length=127

test:
	pytest components

black:
	black --check components

train-nlu:
	python -m rasa train nlu

check: flake black test train-nlu
