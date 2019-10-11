REM Build and install easy_agents as a package on PyPI

cd ..
rmdir /s /q dist
rmdir /s /q easy_agents.egg-info
for /f %%i in ('dir /a:d /s /b env_master\Lib\site-packages\easyagents*') do rmdir /s /q %%i
for /f %%i in ('dir /a:d /s /b env_huskarl\Lib\site-packages\easyagents*') do rmdir /s /q %%i
for /f %%i in ('dir /a:d /s /b env_setup\Lib\site-packages\easyagents*') do rmdir /s /q %%i
pip install twine
python setup.py sdist --formats=zip

pip install pytest

cd dist
pip install easy_agents*
