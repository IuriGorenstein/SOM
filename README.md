# Self Organizing Maps (SOM)
Repository containing code examples on how to reproduce the study of the Atlantic Ocean decadal variability and control over Northeast Brazil and West Africa precipitation using SOM (SOMpy library)

This codes are an adaptation from the SOMpy library. They capture the idea behind the results achieved in the "The Northeast Brazil and West Africa decadal precipitation anti-correlation" article (currently in review). They are a part of my masters degree in Physical Ocenography project and dissertation, at Universidade de  São Paulo, Brazil.

The example in Drive_AtlanticExample.py uses public data from the Cmip6 climate models to create the Self Organizing Maps feature space and its clusterization.
The main effort from this code is to compute the percentages matrix, which semmed necessary for the Best Matching Unit correlation since the SOMpy library didn't seem to project the data into the feature space and back to the input space to describe the climate data as desired.

This code uses numpy, xarray, sklearn, carotpy, matplotlib, sompy, sys, copy, pandas and time libraries.
The example using the Cmip6 Atlantic ocean sea surface temperatures dataset needs xesmf, zarr, fsspec and gcsfs libraries to run. However, the functions in model.py and tools.py work with any other preprocessed dataset. To exemplify its usage there is another example that creates a vibrating pendulum dataset for training and testing.

Many functions are an adaptation from other libraries, such as sompy, and may contain some comments in portuguese.

If you are a fellow scientist or researcher, feel free to use and adapt this code. If you do, please cite us in your article.

 Iuri Gorenstein.
