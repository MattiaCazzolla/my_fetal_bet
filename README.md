# My Fetal BET  
A customized version of inference script of [**fetal-BET**](https://github.com/bchimagine/fetal-brain-extraction) with enhanced functionality and code improvements.  

## Key Differences  
- Support for 4D data  
- Improved code structure and organization  
- File-based input (no longer requires only folder input)  
- Customizable output file suffix  


## Requirements  

To install the necessary dependencies, run:  
```bash
pip install -r requirements.txt
```

## Weights Download

**Download the singularity file:**
```
singularity pull fetal-bet.sif docker://faghihpirayesh/fetal-bet
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

## Example 1 (file inference)
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

## Example 2 (folder inference)
```bash
python inference.py --input_path /data/sub-001 \
                    --output_path /data/sub-001_mask/ \
                    --saved_model_path /code/fetal-bet/AttUNet.pth \
                    --suffix fetalbet_mask
```

The output will be:
```bash
 data/
   ├──sub-001/
   |    ├── T2_run_1.nii.gz 
   |    └── T2_run_2.nii.gz 
   |
   └──sub-001_mask/    
        ├── T2_run_1_fetalbet_mask.nii.gz 
        └── T2_run_2_fetalbet_mask.nii.gz 
```
