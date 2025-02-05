# My Fetal BET  
A customized version of [**fetal-BET**](https://github.com/bchimagine/fetal-brain-extraction) with enhanced functionality and code improvements.  

## Key Features  
- Support for 4D data  
- Improved code structure and organization  
- File-based input (no longer requires folder input)  
- Customizable output file suffix  


## Requirements  

To install the necessary dependencies, run:  
```bash
pip install -r requirements.txt
```

## Weights Download

**Download the singularity file:**
```
singularity pull faghihpirayesh/fetal-bet
```

**Copy the weights to local host:**

```
singularity exec /path/to/fetal-bet.sif cp /app/src/saved_models/AttUNet.pth .
```


## Usage

```bash
python inference.py --input_path /path/to/input \
                    --output_path /path/to/output_folder \
                    --saved_model_path /path/to/weights \
                    --suffix mask
```

## Example
```bash
python inference.py --input_path /data/sub-001/T2.nii.gz \
                    --output_path /data/sub-001/ \
                    --saved_model_path /code/fetal-bet/AttUNet.pth \
                    --suffix fetalbet_mask
```

The output will be:
```bash
 data/
   └──sub-001/
        ├── T2.nii.gz 
        └── T2_fetalbet_mask.nii.gz 
```

