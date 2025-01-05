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
## Acknowledgements
This project was done as an internship as part of the Master in Space Science and Technology at the Observatoire de Paris. The internship took place at LESIA, Observatoire de Paris-Meudon under the supervision of Dr. Pierre Kervella. This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (project UniverScale, grant agreement 951549).
