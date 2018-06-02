# Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images
This folder contains the data.

## Change to custom data location
If your data is saved in a different directory, no worries. You can pass the path of the directory and the files will load the data from the specified directory location.

```
python main.py --data_dir <data directory location>
```

Please make sure that your directory contains the **train.txt** and **val.txt** files. Our code expects the names of images in a particular format

```
<RGB IMAGE>, <LABEL IMAGE>, <Diagnostic Label>
```

For examples, please see **train.txt** and **val.txt** files.
