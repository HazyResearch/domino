autoformat:
	black domino/ tests/
	autoflake --in-place --remove-all-unused-imports -r domino	tests
	isort --atomic domino/ tests/
	docformatter --in-place --recursive domino tests

lint:
	isort -c domino/ tests/
	black domino/ tests/ --check
	flake8 domino/ tests/

test:
	pytest

test-basic:
	set -e
	python -c "import domino"
	python -c "import domino.version as mversion"

test-cov:
	pytest --cov=./ --cov-report=xml

docs:
	sphinx-build -b html docs/source/ docs/build/html/

docs-check:
	sphinx-build -b html docs/source/ docs/build/html/ -W

livedocs:
	sphinx-autobuild -b html docs/source/ docs/build/html/

dev:
	pip install black isort flake8 docformatter pytest-cov sphinx-rtd-theme nbsphinx recommonmark pre-commit

all: autoformat lint docs test
