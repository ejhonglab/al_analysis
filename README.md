
#### Installation

Only tested on Ubuntu 18.04. Python 3.7 and earlier seemed to have some
incompatibilities with `suite2p`.

```
sudo apt update
# TODO also install necessary apt packages for qt5 backend of pyqt5
sudo apt install python3.8-minimal python3.8-venv

python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip

git clone https://github.com/ejhonglab/hong2p ../hong2p
pip install -e ../hong2p

pip install suite2p
```

