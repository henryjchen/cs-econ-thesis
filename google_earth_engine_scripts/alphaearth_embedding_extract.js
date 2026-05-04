/*
This script extracts average satellite embedding features (AlphaEarth 2020)
for a set of polygons (e.g., census units in Mexico).

Because the dataset is large and high-dimensional (64 bands), the script:
- Splits the polygons into smaller chunks to avoid memory limits
- Computes mean embedding values per polygon using reduceRegions
- Removes geometries to keep output files small
- Exports each chunk as a separate CSV to Google Drive

You can adjust `chunkSize` depending on memory constraints.
*/

// Load the polygon dataset (joined census features)
var polygons = ee.FeatureCollection("projects/mexico-census/assets/all_mexico_merged_2020");

// Load AlphaEarth 2020 annual embeddings and combine into a single image
var embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                  .filter(ee.Filter.date('2020-01-01', '2020-12-31'))
                  .mosaic();

// Set chunk size to avoid "User Memory Limit Exceeded" errors
var chunkSize = 750;

// Get total number of features and convert to a list for slicing
var totalFeatures = polygons.size().getInfo();
var agebList = polygons.toList(totalFeatures);

// Loop over the dataset in chunks
for (var i = 0; i < totalFeatures; i += chunkSize) {
  var chunkIndex = Math.floor(i / chunkSize);
  var subset = ee.FeatureCollection(agebList.slice(i, i + chunkSize));
  
  // Compute mean embedding values for each polygon in the subset
  var reduced = embeddings.reduceRegions({
    collection: subset,
    reducer: ee.Reducer.mean(),
    scale: 10,      // Native resolution of AlphaEarth embeddings
    tileScale: 16   // Helps prevent memory issues for large datasets
  });

  // Remove geometries to reduce CSV file size
  var exportTable = reduced.map(function(f) {
    return f.setGeometry(null);
  });

  // Export results to Google Drive as a CSV
  Export.table.toDrive({
    collection: exportTable,
    description: 'mexico_2020_embeddings_' + chunkIndex,
    folder: 'all_mexican_pop_density_prediction',
    fileFormat: 'CSV'
  });
}