# Self Organizing Maps (SOM)
Codes used in study involving the Atlantic Ocean decadal variability and control over NE and WAF precipitation (with SOMpy)

This codes are an adaptation from the SOMpy library, used to achieve the results in the "Northeast Brazil and West Africa decadal precipitation anti-correlation patterns: an exploration of the decadal Atlantic variability using non supervised neuron-networks" article (currently in review). They are part of Iuri Gorenstein's masters degree in Physical Ocenography project and dissertation, in Universidade de  São Paulo, Brazil.

The example in Drive.py uses public data from the Cmip6 climate models to create the Self Organizing Maps feature space and its clusterization.
The main effort from this code is to compute the percentages matrix, which semmed necessary for the Best Matching Unit correlation since the SOMpy library didn't seem to project the data in to the feature space and back to the input space to describe the climate data as desired.

This code uses numpy, xarray, sklearn, carotpy, matplotlib, sompy, sys, copy, pandas and time libraries
The example using the Cmip6 dataset needs xesmf, zarr, fsspec and gcsfs libraries to run. However, the functions work with any other preprocessed dataset.

Many functions are an adaptation from other libraries such as sompy and may contain some comments in portuguese.

If you are a fellow scientist or researcher, feel free to use and adapt this code. If you do, please cite us in your article.


Iuri Gorenstein.
