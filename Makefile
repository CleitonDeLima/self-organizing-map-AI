install:
	sudo apt-get install python3-pip python3-venv tk-dev
	pip3 install -r requirements.txt
	pip3 install "ipython[notebook]"

run:
	jupyter-notebook execucao-programa.ipynb