REM Build and install easy_agents as a package on PyPI

cd ..
rmdir /s /q dist
rmdir /s /q easy_agents.egg-info
for /f %%i in ('dir /a:d /s /b env\Lib\site-packages\easyagents*') do rmdir /s /q %%i
pip install twine
python setup.py sdist --formats=zip
REM python setup.py sdist bdist_wheel
twine check dist/*
REM twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*