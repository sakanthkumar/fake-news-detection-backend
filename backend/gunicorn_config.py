# gunicorn_config.py
workers = 1
threads = 1
timeout = 120
preload_app = False
max_requests = 1000
