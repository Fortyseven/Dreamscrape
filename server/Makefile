dev:
	#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24 CUDA_LAUNCH_BLOCKING=1
	# python -m flask run
	python ./app.py
debug:
	PYTHONBREAKPOINT="pudb.set_trace" python -m pudb -m flask --debug --app server.py run
