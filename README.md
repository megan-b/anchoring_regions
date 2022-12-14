This repository supports the paper titled "Identifying individual anchoring regions
by mining public transport smart card data" and contains additional supporting charts and information as an appendix to the paper.


## Using this code

Two datasets are required: journeys (containing the tag on and off transactions for a public transport network) and stops (the list of stop IDs, as used in the journeys data, and their corresponding names and coordinates).

For confidentiality reasons the original data cannot be provided. A synthetic dataset will instead be provided for use, but note the results will not be exactly reproducible.

Two other minor updates to the code will be required in order for it to run - a directory string to where the data is saved, and a Mapbox token to generate the maps. Mapbox provides further information on generating access tokens [[link]](https://docs.mapbox.com/help/getting-started/access-tokens/ "here").

## Supporting charts

Since the results are not directly reproducible, throughout the code there are references to figure numbers and the plots generated from this code with the original data are included as static images here.

![Image](./assets/readme_fig1.PNG "Figure 1")
Figure 1: Histogram of number of card uses (with no trips dropped) - used to determine threshold for card inclusion in analysis

![Image](./assets/readme_fig2.PNG "Figure 2")
Figure 2: Example plot of stop clusters - showing where stops close by are considered together

![Image](./assets/readme_fig3.PNG "Figure 3")
Figure 3: Histogram of distances in stays (distance between disembarking a vehicle and embarking the next)

![Image](./assets/readme_fig4.png "Figure 4")
Figure 4: Number of visits to each identified region - used to identify where a region should be considering 'anchoring' vs 'visited'

![Image](./assets/readme_fig4a.png "Figure 4a")
Figure 4a: Zoomed in version of Figure 4 due to the long tail

![Image](./assets/readme_fig5a.PNG "Figure 5a")
Figure 5a: Fraction of time covered by anchoring region (what % of the time in the month does the passenger spend in anchoring regions?), for passengers with at least one anchoring region 

![Image](./assets/readme_fig5b.PNG "Figure 5b")
Figure 5b: Fraction of visits covered by anchoring region (what % of the visits in the month are to anchoring regions?), for passengers with at least one anchoring region 

![Image](./assets/readme_fig6.PNG "Figure 6")
Figure 6: Chart of AIC and BIC vs number of clusters, used to identify the number of clusters to be used

![Image](./assets/readme_fig7.PNG "Figure 7")
Figure 7: Number of anchoring regions per passenger

![Image](./assets/readme_fig8.PNG "Figure 8")
Figure 8: Activity fractions by region cluster (what is the distribution of activity types for each region that has been identified as being in this region type?)

![Image](./assets/readme_fig9.PNG "Figure 9")
Figure 9: Land use fractions by region cluster (what is the distribution of land use types for each region that has been identified as being in this region type?)

![Image](./assets/readme_fig10.PNG "Figure 10")
Figure 10: Number of 'Residence' type anchoring regions per passenger

![Image](./assets/readme_fig11.PNG "Figure 11")
Figure 11: Region centroids on a map, coloured by region cluster. Visual inspection / local knowledge allows checking of region identification against local landmarks and points of interest. Note that 'Residence' regions have been excluded from this map for clarity.
