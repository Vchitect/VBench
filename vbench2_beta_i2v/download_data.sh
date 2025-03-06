mkdir -p vbench2_beta_i2v/data
gdown --id 1zmWs_m_A4q6YgTZwIZ230jW0ttknlGJA --output vbench2_beta_i2v/data/i2v-bench-info.json
gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ --output vbench2_beta_i2v/data/crop.zip
gdown --id 1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY --output vbench2_beta_i2v/data/origin.zip
unzip vbench2_beta_i2v/data/crop.zip -d vbench2_beta_i2v/data
unzip vbench2_beta_i2v/data/origin.zip -d vbench2_beta_i2v/data
rm -f vbench2_beta_i2v/data/crop.zip
rm -f vbench2_beta_i2v/data/origin.zip