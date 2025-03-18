wget -O data.zip https://cloud.tsinghua.edu.cn/f/431c286e5e7c42c5aebf/?dl=1
unzip data.zip
mv A4-data/* ./
rm -rf A4-data/
rm -rf data.zip