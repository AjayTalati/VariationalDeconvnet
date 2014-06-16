sudo yum install -y gcc-gfortran
sudo yum install -y git

# Get and build OpenBlas (Torch is much better with a decent Blas)
cd /tmp/
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make NO_AFFINITY=1 USE_OPENMP=1
RET=$?; if [ $RET -ne 0 ]; then
    echo "Error. OpenBLAS could not be compiled";
    exit $RET;
fi
sudo make install
RET=$?; if [ $RET -ne 0 ]; then
    echo "Error. OpenBLAS could not be compiled";
    exit $RET;
fi

PREFIX=${PREFIX-/usr/local}

export CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include/:/opt/OpenBLAS/lib/:/usr/lib/gcc/x86_64-amazon-linux:/usr/lib/gcc/x86_64-amazon-linux/4.8.2:$CMAKE_LIBRARY_PATH

export CUDA_BIN_PATH=/opt/nvidia/cuda/bin

# Install base packages:
sudo -E $PREFIX/bin/luarocks install sundown
sudo -E $PREFIX/bin/luarocks install cwrap  
sudo -E $PREFIX/bin/luarocks install paths  
sudo -E $PREFIX/bin/luarocks install torch  
sudo -E $PREFIX/bin/luarocks install nn     
# $PREFIX/bin/luarocks install dok           ||  sudo -E $PREFIX/bin/luarocks install dok    
# $PREFIX/bin/luarocks install gnuplot       ||  sudo -E $PREFIX/bin/luarocks install gnuplot

sudo -E $PREFIX/bin/luarocks install cutorch
sudo -E $PREFIX/bin/luarocks install cunn   

# $PREFIX/bin/luarocks install qtlua         ||  sudo -E $PREFIX/bin/luarocks install qtlua  
# $PREFIX/bin/luarocks install qttorch       ||  sudo -E $PREFIX/bin/luarocks install qttorch
sudo -E $PREFIX/bin/luarocks install luafilesystem
sudo -E $PREFIX/bin/luarocks install penlight 
sudo -E $PREFIX/bin/luarocks install sys      
sudo -E $PREFIX/bin/luarocks install xlua     
# $PREFIX/bin/luarocks install image         ||  sudo -E $PREFIX/bin/luarocks install image    
sudo -E $PREFIX/bin/luarocks install optim    
sudo -E $PREFIX/bin/luarocks install lua-cjson
sudo -E $PREFIX/bin/luarocks install trepl    
