# DataAugmentation directory

```bash
ln -s /home/alissonchiquitto/work/DataAugmentation DataAugmentation
```

# Criar o ambiente CONDA

```bash
conda create python=3.10.9 --name HumanPreMiRNA -y

conda activate HumanPreMiRNA

pip install --upgrade keras
pip install tensorflow
pip install pandas
pip install scikit-learn
python -m pip install viennarna
pip install tensorflow[and-cuda]
```

Training a CNN:

```bash
cd human_pre_miRNA/CNN
conda activate HumanPreMiRNA
python CNNTrain_chiquitto.py --pos ../dataset/pos.csv --neg ../dataset/neg.csv --output ../models
```

Classifying sequences with CNN:

```bash
cd human_pre_miRNA/CNN
conda activate HumanPreMiRNA
python CNNEvaluation_chiquitto.py --input ../dataset/pos.csv --model ../models/CNN_model.h5 --output ../models/output.csv
```

Training a RNN:

```bash
cd human_pre_miRNA/RNN
conda activate HumanPreMiRNA
python RNNTrain_chiquitto.py --pos ../dataset/pos.csv --neg ../dataset/neg.csv --output ../models
```

Classifying sequences with RNN:

```bash
cd human_pre_miRNA/CNN
conda activate HumanPreMiRNA
python RNNEvaluation_chiquitto.py --input ../dataset/pos.csv --model ../models/RNN_model.h5 --output ../models/output.csv
```
