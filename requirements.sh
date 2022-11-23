# pip install fairscale
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale 
git checkout v0.4.6
# git am --abort
git am --signoff < ../patch/0002-Modified-MOELayer.patch
# git am --signoff < ../patch/0001-0.4.6-release.patch

pip install timm

pip install yacs
pip install pytest
pip install termcolor
cd ..
git+https://github.com/microsoft/tutel@main
python3 -m pip uninstall tutel -y
python3 ./tutel/setup.py install --user


# pip install jupyterlab