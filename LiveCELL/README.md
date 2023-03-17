## Probabilistic Domain Adaptation on [LiveCELL](https://www.nature.com/articles/s41592-021-01249-6) Dataset

> Dataset is publicly available [here](https://sartorius-research.github.io/LIVECell/)

Implements the domain adaptation experiments on the LiveCELL dataset. Each script implements training, prediction and validation of the respective model.
All the required data can be downloaded with the script `prepare_data.py`.

For more information on the individual scripts refer to their help output, e.g. `python livecell_unet.py -h`.


### livecell_unet.py (Source UNet Training)
```
python livecell_unet.py [--train / --predict / --evaluate]
                        --data <PATH-TO-LIVECELL-DATA>
                        [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### livecell_punet.py (Source PUNet Training)
```
python livecell_punet.py [--train / --predict / --evaluate]
                         --data <PATH-TO-LIVECELL-DATA>
                         [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### livecell_punet_target.py (Target PUNet Training using Pseudo Labels from Source)
```
python livecell_punet_target.py --get_pseudo_labels
                                [--train / --predict / --evaluate]
                                [(optional : allows consensus masking) --consensus]
                                --data <PATH-TO-LIVECELL-DATA>
                                [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### livecell_mt.py (Mean-Teacher Separate Training)
```
python livecell_mt.py [--train / --predict / --evaluate]
                      [(optional : enables consensus weighting) --consensus]
                      [(optional : enables consensus masking) --consensus --masking]
                      --data <PATH-TO-LIVECELL-DATA>
                      [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```


### livecell_fm.py (FixMatch Separate Training)
```
python livecell_fm.py [--train / --predict / --evaluate]
                      [(optional : enables consensus weighting) --consensus]
                      [(optional : enables consensus masking) --consensus --masking]
                      --data <PATH-TO-LIVECELL-DATA>
                      [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```


### livecell_adamatch.py (FixMatch-based Joint Training)
```
python livecell_adamatch.py [--train / --predict / --evaluate]
                            [(optional : enables consensus weighting) --consensus]
                            [(optional : enables consensus masking) --consensus --masking]
                            --data <PATH-TO-LIVECELL-DATA>
                            [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```

### livecell_adamt.py (Mean-Teacher-based Joint Training)
```
python livecell_adamt.py [--train / --predict / --evaluate]
                         [(optional : enables consensus weighting) --consensus]
                         [(optional : enables consensus masking) --consensus --masking]
                         --data <PATH-TO-LIVECELL-DATA>
                         [(optional) --pred_path <PATH-TO-SAVE-PREDICTIONS>]
```
