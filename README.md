Use python 3.10 

We're adding os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none' because of a Streamlit + PyTorch compatibility issue because Streamlit's file watcher tries to inspect all loaded modules including PyTorch which doesn't allow it.