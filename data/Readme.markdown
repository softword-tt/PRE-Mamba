The data can be placed under this directory. Taking SYTHETIC_DATA as an example, the data is divided and processed according to time windows into the following directory structure:

```
data/
└── synthetic/
    └── SYNTHETIC_DATA_DIR/
        └── synthetic/
            ├── merge_data/
            │   ├── 1mm/
            │   │   ├── 0000000000.npz
            │   │   ├── ...
            │   │   └── 0000000388.npz
            │   ├── 5mm/
            │   │   ├── 0000000000.npz
            │   │   ├── ...
            │   │   └── 0000000388.npz
            │   ├── 10mm/
            │   │   └── ...
            │   ├── ...
            │   └── 200mm/
            │       ├── 0000000000.npz
            │       ├── ...
            │       └── 0000000388.npz
            └── raw_data/
                ├── 0000000000.npz
                ├── ...
                └── 0000000388.npz
└── real/
    └── EVK4_DATA_DIR/              # EVK4_artificial and EVK4_realworld
        └── scene{i}/               # Take scene1 as an example
            ├── merge_data/
            │   ├── rain_0/
            │   │   ├── 0000000000.npz
            │   │   ├── ...
            │   │   └── 0000000165.npz
            │   ├── rain_1/
            │   │   ├── 0000000000.npz
            │   │   ├── ...
            │   │   └── 0000000182.npz
            │   ├── rain_2/
            │   │   └── ...
            │   ├── ...
            │   └── rain_3/
            │       ├── 0000000000.npz
            │       ├── ...
            │       └── 0000000166.npz
            └── labels/
                │── labels_rain_0/
                │   ├── labels_0000000000.npz
                │   ├── ...
                │   └── labels_0000000165.npz
                ├── labels_rain_1/
                │   ├── labels_0000000000.npz
                │   ├── ...
                │   └── labels_0000000182.npz
                ├── labels_rain_2/
                │   └── ...
                ├── ...
                └── labels_rain_3/
                    ├── labels_0000000000.npz
                    ├── ...
                    └── labels_0000000166.npz
```