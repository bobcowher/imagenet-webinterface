apt-get install python3
apt-get install python3-pip
pip install flask
pip install wtforms
pip install torch --no-cache-dir
apt-get install git


git clone https://github.com/bobcowher/imagenet-webinterface.git

nohup python main.py > flask.log 2>&1 &