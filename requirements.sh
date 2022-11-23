# pip install fairscale
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale 
git tag 0.4.6

pip install timm

pip install yacs
pip install pytest
pip install termcolor
git+https://github.com/microsoft/tutel@main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py install --user


# pip install jupyterlab