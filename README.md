# BIA_group_work
The repository for BIA ICA


## Repo Structure
BIA_group_work/              
├── requirements.txt          
├── README.md                 # project description
│
├── gui/                      # GUI interface
│
├── prediction/               # Use trained model for prediction
│
├── preprocessing/            # Image segmentation
│
├── trained_models/           
│   ├── unet_model.pth   
│   ├── mobilenet_based_model.pth   
│   └── efficientnet_based_model.pth            
│
└── notebook/                 # train and test process
    ├── train.ipynb
    └── test_segmentation.ipynb
