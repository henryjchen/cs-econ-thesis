/*
This script creates cloud-free Landsat 7 surface reflectance composites
for Mexico in 2020 using two different masking approaches:

1) A simple cloud + shadow mask using QA bits
2) A more complete Landsat 4/5/7 SR mask that also:
   - Applies radiometric scaling
   - Masks saturated pixels

It then:
- Builds median composites for each approach
- Displays the results on the map for comparison
- Exports the enhanced (scaled + masked) composite as an Earth Engine asset
*/


// --- Cloud masking (simple QA-based mask) ---
function maskLandsatSR(image) {
    var qa = image.select('QA_PIXEL');
  
    // Bit flags for cloud shadow and clouds
    var cloudShadowBitMask = 1 << 3;
    var cloudBitMask = 1 << 5;
  
    // Keep pixels that are NOT cloud or shadow
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudBitMask).eq(0));
  
    return image.updateMask(mask);
  }
  
  
  // --- Cloud masking + scaling for Landsat 4/5/7 SR ---
  function maskL457sr(image) {
    // Mask out fill, cloud, shadow, etc. using QA bits
    var qaMask = image.select('QA_PIXEL')
                      .bitwiseAnd(parseInt('11111', 2))
                      .eq(0);
  
    // Mask saturated pixels
    var saturationMask = image.select('QA_RADSAT').eq(0);
  
    // Apply scale factors to optical bands
    var opticalBands = image.select('SR_B.')
                            .multiply(0.0000275)
                            .add(-0.2);
  
    // Apply scale factors to thermal band
    var thermalBand = image.select('ST_B6')
                           .multiply(0.00341802)
                           .add(149.0);
  
    // Replace bands with scaled versions and apply masks
    return image.addBands(opticalBands, null, true)
                .addBands(thermalBand, null, true)
                .updateMask(qaMask)
                .updateMask(saturationMask);
  }
  
  
  // --- Define Mexico boundary ---
  var mexico = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                .filter(ee.Filter.eq('country_na', 'Mexico'))
                .geometry();
  
  
  // --- Visualization settings ---
  var visParams = {
    bands: ['SR_B3', 'SR_B2', 'SR_B1'],
    min: 5000,
    max: 15000
  };
  
  var visParams2 = {
    bands: ['SR_B3', 'SR_B2', 'SR_B1'],
    min: 0,
    max: 0.25
  };
  
  
  // --- Unmasked median composite (baseline) ---
  var unmasked = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                  .filterDate('2020-01-01', '2020-12-31')
                  .filterBounds(mexico)
                  .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
                  .median()
                  .clip(mexico);
  
  
  // --- Composite with simple cloud masking ---
  var masked = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                .filterDate('2020-01-01', '2020-12-31')
                .filterBounds(mexico)
                .map(maskLandsatSR)
                .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
                .median()
                .clip(mexico);
  
  
  // --- Composite with enhanced masking + scaling ---
  var masked2 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                 .filterDate('2020-01-01', '2020-12-31')
                 .filterBounds(mexico)
                 .map(maskL457sr)
                 .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7'])
                 .median()
                 .clip(mexico);
  
  
  // --- Display results ---
  Map.centerObject(mexico, 6);
  // Map.addLayer(unmasked, visParams, 'Unmasked Composite'); // optional baseline
  Map.addLayer(masked, visParams, 'Cloud-Masked Composite');
  Map.addLayer(masked2, visParams2, 'Masked (Enhanced)');
  
  
  // --- Export enhanced composite to Earth Engine asset ---
  Export.image.toAsset({
    image: masked2,
    description: 'cloud_masked_mexico_enhanced_2020',
    assetId: 'projects/mexico-census/assets/cloud_masked_mexico_enhanced_2020',
    region: mexico,
    scale: 30,
    maxPixels: 1e13
  });
  
  
  // Inspect final image in console
  print(masked2);