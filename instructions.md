Repository setup:
1) Create folder at chosen destination, name it "netid-a2"
2) Clone repository into folder, the dataset will also go into this folder, but not into the repository
3) Once the repository is cloned, load the conda module and follow these commands
    cd a2-<your-netid>
    conda env create -f environment.yml -n a2
    conda activate a2 

Dataset Setup: 
1) Download dataset
2) Create dir = "data/pamap2"
3) Select the subjectxxx.dat from protocol
4) Run dataset_preprocessing.py to get train/val/test splits

Training setup: 
1) Once the environment is setup, run training using- 
    for fusion in early late hybrid uncertainty; do
        python src/train.py model.fusion_type=$fusion
    done
2) This will start training for all three fusion models

Testing/Analysis:
1) How to run missing modality test python src/eval.py --checkpoint runs/a2_hybrid_pamap2/best.ckpt --missing_modality_test
2) How to run fusion test
    - Create fusion_compare.json by running compare_ckpts.py
    - Run analysis.py with 
        python src/analysis.py \                                                                                                       
        --experiment_dir experiments \
        --output_dir analysis \

3) To get attention matrix, run val_attention.py
    - this will get the attention matrix from the best training log and output attn.png
        python src/analysis.py \                                                                                                      
        --experiment_dir experiments \
        --output_dir analysis \
        --attn_npy experiments/attn/attn_MxM.npy

Uncertainty aware training information
1) Handled with training setup, will output uncertainty.json

Ablation Studies:
1) Run python -W ignore src/train.py \
    model.fusion_type=hybrid \
    model.num_heads=8 \
    experiment.save_dir=./runs/ablation-8head \
    outputs.experiments_dir=./experiments/ablation-8head \
    outputs.analysis_dir=./analysis/ablation-8head \
    experiment.name="ablation_${fusion}"

