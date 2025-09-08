# gunicorn_config.py

# Increase the timeout for workers to start
timeout = 300

def post_fork(server, worker):
    """
    This function is called in each worker process after it's created.
    We use it to load our ML model, ensuring each worker has its own copy.
    """
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    
    # Import and run our model loading function
    from colorizer import load_model
    load_model()
    server.log.info("ML model loaded in worker (pid: %s)", worker.pid)