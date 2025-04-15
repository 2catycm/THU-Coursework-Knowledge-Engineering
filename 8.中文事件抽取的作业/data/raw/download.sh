wget -O data.zip https://cloud.tsinghua.edu.cn/f/88ef126dfd1d40fe801e/?dl=1
unzip data.zip
mv A8-data/* ./
rm -rf A8-data/
rm -rf data.zip