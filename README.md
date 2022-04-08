# rsb
Experiments for: *Class-Incremental Experience Replay for Continual Learning under Concept Drift*.

### Code
- ER, RSB: *learners/er.py*

- Experiments: *benchmark/er_runner.py*

### Packages
```
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

### Extractors
Can be downloaded from: https://drive.google.com/drive/folders/1bNv8s3QXrZWMy1PTs_4B2TGTeM_HzI21?usp=sharing

### Run
If you use TensorBoard: 

```
tensorboard --logdir runs/er
```

Then:

```
python main.py
```

### Results

- CSV files can be found in *results*.

- TensorBoard: http://localhost:6006/

### Citation

```
@article{Korycki2021:er,
  title={Class-Incremental Experience Replay for Continual Learning under Concept Drift},
  author={Lukasz Korycki and B. Krawczyk},
  journal={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2021},
  pages={3644-3653}
}
```
