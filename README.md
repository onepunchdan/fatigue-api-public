## Environment setup (tested on Ubuntu 16.04, Anaconda 5)
create virtual env with python 3 and keras
- `conda create --name deeplearning -c conda-forge python=3 keras tensorflow sk-video flask pillow gevent requests opencv scipy numpy scikit-learn`

activate virtual env
- `source activate deeplearning`

add some packages not found/updated in conda repo
- `pip install dlib imutils`

## Start server
`python run_keras_server.py`

## Test API response
Need to modify 'IMAGE_PATH' in `simple_request.py` with path to .mp4 file (at least 5 minutes, depends on padding parameter in `run_keras_server.py`)

`python simple_request.py`
