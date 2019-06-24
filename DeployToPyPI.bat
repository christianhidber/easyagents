REM Build and install easy_agents as a package on PyPI

rmdir /s /q dist
rmdir /s /q easy_agents.egg-info
pip install twine
python setup.py sdist --formats=zip
REM python setup.py sdist bdist_wheel
twine check dist/*
REM twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*