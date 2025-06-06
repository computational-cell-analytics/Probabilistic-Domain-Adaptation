# Probabilistic Domain Adaptation on [Lung X-Ray Datasets](https://arxiv.org/abs/1904.09229)

> Datasets are available [here](https://github.com/rsummers11/CADLab/tree/master/Lung_Segmentation_XLSor/data)

Implements the domain adaptation experiments on the lung X-Ray datasets. Each script implements training, prediction and validation of the respective model.
To prepare the experiments download the data from the link provided above, and:
- unzip `XLSor_data.zip`
- unzip the two other datasets, and put them in the corresponding folders in the unzipped xlsor data
- run `prepare_data.py <PATH>` where `<PATH>` points to the unzipped `XLSor_data` folder

For more information on the individual scripts refer to their help output, e.g. `python lung_unet.py -h`.


### lung_unet.py (Source UNet Training)
```
python lung_unet.py [--train / --predict/ --evaluate]
                    --data <PATH-TO-LUNG-XRAYS>
                    --pred_path <PATH-FOR-PREDICTIONS>
```

### lung_punet.py (Source PUNet Training)
```
python lung_punet.py [--train / --predict/ --evaluate]
                     --data <PATH-TO-LUNG-XRAYS>
                     --pred_path <PATH-FOR-PREDICTIONS>
```

### lung_mt.py (Mean-Teacher Training)
```
python lung_mt.py [--train / --predict / --evaluate]
                  [(optional : enables consensus weighting) --consensus]
                  [(optional : enables consensus masking) --consensus --masking]
                  --data <PATH-TO-LUNG-XRAYS>
                  [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### lung_adamt.py (Mean-Teacher-based Joint Training)
```
python lung_adamt.py [--train / --predict / --evaluate]
                     [(optional : enables consensus weighting) --consensus]
                     [(optional : enables consensus masking) --consensus --masking]
                     --data <PATH-TO-LUNG-XRAYS>
                     [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```
