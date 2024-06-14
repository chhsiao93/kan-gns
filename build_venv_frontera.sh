module rest
module load python3/3.9
module load cuda/12

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-geometric==2.4.0
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install -r requirements.txt

wget -O WaterDropSample.zip https://github.com/kks32-courses/sciml/raw/main/lectures/12-gnn/WaterDropSample.zip
unzip WaterDropSample.zip


mkdir -p temp/models/WaterDropSample
mkdir -p temp/rollouts/WaterDropSample