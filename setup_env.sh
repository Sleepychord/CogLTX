# local
# ssh-copy-id -i ~/.ssh/id_rsa.pub  mingding@192.168.x.xxx (-p 12345)
# vim ~/.ssh/config

echo "Starting Anaconda setup"
echo "Installing Miniconda"
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
echo "Finished installing Miniconda"
conda -y create -n flyback python=3.7
conda init bash

# reopen

source activate flyback
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy torch==1.3.1 torchvision==0.4.2 transformers pytorch-lightning gpustat
ujson fuzzywuzzy