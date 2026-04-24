# Outlier detection

The relationship between all molecules are descriptor/fingerprint and distance metric determined. Infering that outlier detection methods must work on a distance matrix.

## Outlier detection Methods
- HDBSCAN: Extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using a technique to extract a flat clustering based in the stability of clusters. Like DBSCAN it marks some points as noise, in this case hopefully our outliers. It is a commonly used method in outlier detection.
- Local Outlier Factor (LOF): Detects outliers based on local density rather than global distance.
- k Nearest Neighbours: calculates the distance to the k nearest neighbours - the outliers here are the points with the largest distance. 

## Validation
- Recall: Is high the false negative score is low --> when there are few of the original points from qm9 that are flagged as outliers.
- Precision: Measures the accuracy of the correct predictions, does it classify all injected outliers as outliers?