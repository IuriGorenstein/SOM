# Self Organizing Maps (SOM)
Repository containing code examples on how to reproduce the study of the Atlantic Ocean decadal variability and control over Northeast Brazil and West Africa precipitation using SOM (SOMpy library)

This code captures the idea behind the results achieved in "A 50-year cycle of sea surface temperature regulates decadal precipitation in the tropical and South Atlantic region" - Gorenstein et al., (2023). The article's results can be reproduced with the use of the codes hereby, and the data mentioned in the paper. This study is part of my masters degree in Physical Ocenography dissertation at Universidade de São Paulo, Brazil from 2021 to 2023.

The example in Drive_AtlanticExample.py uses public data from the Cmip6 climate models to create the Self Organizing Maps feature space and its clusterization.
The main effort from this code is to compute the percentages matrix, which semmed necessary for the Best Matching Unit correlation since the SOMpy library didn't seem to project the data into the feature space and back to the input space to describe the climate data as desired.

This code uses numpy, xarray, sklearn, carotpy, matplotlib, sompy, sys, copy, pandas and time libraries.
To exemplify its usage without futher libraries, there is an example that creates a vibrating pendulum dataset for training and testing using the functions in model.py and too.py.
The example using the Cmip6 Atlantic ocean sea surface temperatures dataset needs xesmf, zarr, fsspec and gcsfs libraries to run.

Many functions are an adaptation from other libraries, such as sompy, and may contain some comments in portuguese.

If you are a fellow scientist or researcher, feel free to use and adapt this code. If you do, please cite the Gorenstein et al., (2023) paper in your article.

 Iuri Gorenstein.

References:

Gorenstein, I., Wainer, I., Pausata, F.S.R. et al. A 50-year cycle of sea surface temperature regulates decadal precipitation in the tropical and South Atlantic region. Commun Earth Environ 4, 427 (2023). https://doi.org/10.1038/s43247-023-01073-0
