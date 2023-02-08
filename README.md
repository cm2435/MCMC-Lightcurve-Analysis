# Astrophysics transit analysis 

Some undergrad work on implimenting some periodograms and MCMC transit analysis of Kepler-7b exoplanets. Makes for a useful example of bayesian time series forcasting, or just general cosmology. 

## Development 

We recommend **Python 3.7** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-s** The code does **not** work with Python 2.

Using conda would be advised. To rebuild, make a clean environment and run 
````python
pip install -r requirements-dev.txt
````

## Main structure
The main analysis and commentry is implimented in 
````
/ph30016_b/CourseWork_b.ipynb
````

The transit data can be found in the subpath 
````
/ph30016_b/Data
````

as Astropy .fits files. For reference, this is the kepler dump data for exoplanets orbiting Kepler-7b.

