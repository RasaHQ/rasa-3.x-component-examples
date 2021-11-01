flake:
	flake8 components --count --statistics --max-complexity=10 --max-line-length=127

check: flake
	black --check components
