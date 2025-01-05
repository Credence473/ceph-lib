# Cepheid Photometry
A pipeline to perform photometry on a selected population of stars (Cepheids to be specific) in an image of the host galaxy. 

The `cephlib.py` contains the class and the Jupyter notebooks show example use cases.
In next phase, I may comment the necessary steps in the Jupyter notebook but for now it should be readable.

# Required Libraries
On top of a full Anaconda installation, the following packages should be installed. Creating a separate environment is recommended.
- `webbpsf`
- `photutils`

A full list of all required libraries are: 

- **Core Libraries**:
  - `warnings`
  - `os`
  - `glob`
  - `pathlib`
  - `time`
  - `tarfile`
  - `urllib`

- **Data Handling**:
  - `pandas`
  - `numpy`

- **Visualization**:
  - `matplotlib`
  - `mpl_toolkits`

- **Astrophysics**:
  - `astropy`
  - `webbpsf`
  - `photutils`

- **Scientific Computing**:
  - `scipy`

- **Progress Tracking**:
  - `tqdm`

