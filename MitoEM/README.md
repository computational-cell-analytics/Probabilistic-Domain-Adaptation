# Probabilistic Domain Adaptation of MitoEM Datasets
> Datasets can be found here: [MitoEM](https://mitoem.grand-challenge.org/), [Lucchi](https://casser.io/connectomics/), [VNC](https://github.com/unidesigner/groundtruth-drosophila-vnc), [UroCell](https://github.com/MancaZerovnikMekuc/UroCell)

Implements the domain adaptation experiments on the Mitochondria EM datasets. Each script implements training, prediction and validation of the respective model. The required target domain datasets can be downloaded with the script `prepare_data.py`.

For more information on the individual scripts refer to their help output, e.g. `python mitoem_unet.py -h`.

### mitoem_unet.py (Source UNet Training)
```
python mitoem_unet.py [--train / --predict / --evaluate]
                      --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                      [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### mitoem_punet.py (Source PUNet Training)
```
python mitoem_punet.py [--train / --predict / --evaluate]
                       --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                       [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### mitoen_mt.py (Mean-Teacher Separate Training)
```
python mitoem_mt.py [--train / --predict / --evaluate]
                      [(optional : enables consensus weighting) --consensus]
                      [(optional : enables consensus masking) --consensus --masking]
                      --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                      [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```


### mitoem_fm.py (FixMatch Separate Training)
```
python mitoem_fm.py [--train / --predict / --evaluate]
                      [(optional : enables consensus weighting) --consensus]
                      [(optional : enables consensus masking) --consensus --masking]
                      --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                      [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```


### mitoem_adamatch.py (FixMatch-based Joint Training)
```
python mitoem_adamatch.py [--train / --predict / --evaluate]
                            [(optional : enables consensus weighting) --consensus]
                            [(optional : enables consensus masking) --consensus --masking]
                            --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                            [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### mitoem_adamt.py (Mean-Teacher-based Joint Training)
```
python mitoem_adamt.py [--train / --predict / --evaluate]
                         [(optional : enables consensus weighting) --consensus]
                         [(optional : enables consensus masking) --consensus --masking]
                         --data <PATH-TO-MITOEM/LUCCHI/VNC/UROCELL-DATA>
                         [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```
