@echo off
mkdir vbench2_beta_i2v\data 2>nul
gdown --id 1zmWs_m_A4q6YgTZwIZ230jW0ttknlGJA --output ./vbench2_beta_i2v\data\i2v-bench-info.json
gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ --output ./vbench2_beta_i2v\data\crop.zip
gdown --id 1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY --output ./vbench2_beta_i2v\data\origin.zip
tar -xf ./vbench2_beta_i2v\data\crop.zip -C ./vbench2_beta_i2v\data
tar -xf ./vbench2_beta_i2v\data\origin.zip -C ./vbench2_beta_i2v\data
del /f ./vbench2_beta_i2v\data\crop.zip
del /f vbench2_beta_i2v\data\origin.zip
