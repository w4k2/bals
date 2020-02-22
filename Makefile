all:
	#python -W ignore experiment1.py
	python -W ignore experiment2.py
	#python -W ignore gather1.py
	python -W ignore gather2.py
	#python -W ignore analyze1.py
	python -W ignore analyze2.py
	#./megaplot.sh

clean:
	rm -rf results/*
	rm -rf scores*.npy
	rm -rf plots/*
