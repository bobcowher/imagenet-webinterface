apt-get install python3
apt-get install python3-pip
pip install flask
pip install wtforms
pip install torch --no-cache-dir
pip install numpy
pip install opencv-python
pip install torchvision
apt install libgl1-mesa-glx
apt-get install git

ufw allow 22/tcp
git clone https://github.com/bobcowher/imagenet-webinterface.git

nohup python main.py > flask.log 2>&1 &