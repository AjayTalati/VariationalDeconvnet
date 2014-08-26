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

# Build and install Torch7
cd /tmp
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir build; cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install || sudo -E make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi

# Statuses:
sundown=ok
cwrap=ok
paths=ok
torch=ok
nn=ok
dok=ok
gnuplot=ok
#qtlua=ok
#qttorch=ok
lfs=ok
penlight=ok
sys=ok
xlua=ok
image=ok
optim=ok
cjson=ok
trepl=ok

path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then  
    cutorch=ok
    cunn=ok
fi

# Install base packages:
$PREFIX/bin/luarocks install sundown       ||  sudo -E $PREFIX/bin/luarocks install sundown
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install cwrap         ||  sudo -E $PREFIX/bin/luarocks install cwrap  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install paths         ||  sudo -E $PREFIX/bin/luarocks install paths  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install torch         ||  sudo -E $PREFIX/bin/luarocks install torch  
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install nn            ||  sudo -E $PREFIX/bin/luarocks install nn     
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install dok           ||  sudo -E $PREFIX/bin/luarocks install dok    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install gnuplot       ||  sudo -E $PREFIX/bin/luarocks install gnuplot
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi

[ -n "$cutorch" ] && \
($PREFIX/bin/luarocks install cutorch      ||  sudo -E $PREFIX/bin/luarocks install cutorch        ||   cutorch=failed )
[ -n "$cunn" ] && \
($PREFIX/bin/luarocks install cunn         ||  sudo -E $PREFIX/bin/luarocks install cunn           ||   cunn=failed )

#$PREFIX/bin/luarocks install qtlua         ||  sudo -E $PREFIX/bin/luarocks install qtlua  
#RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
#$PREFIX/bin/luarocks install qttorch       ||  sudo -E $PREFIX/bin/luarocks install qttorch
#RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install luafilesystem ||  sudo -E $PREFIX/bin/luarocks install luafilesystem
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install penlight      ||  sudo -E $PREFIX/bin/luarocks install penlight 
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install sys           ||  sudo -E $PREFIX/bin/luarocks install sys      
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install xlua          ||  sudo -E $PREFIX/bin/luarocks install xlua     
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install image         ||  sudo -E $PREFIX/bin/luarocks install image    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install optim         ||  sudo -E $PREFIX/bin/luarocks install optim    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install lua-cjson     ||  sudo -E $PREFIX/bin/luarocks install lua-cjson
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
$PREFIX/bin/luarocks install trepl         ||  sudo -E $PREFIX/bin/luarocks install trepl    
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi

# Done.
echo ""
echo "=> Torch7 has been installed successfully"
echo ""
echo "  + Extra packages have been installed as well:"
echo "     $ luarocks list"
echo ""
echo "  + To install more packages, do:"
echo "     $ luarocks search --all"
echo "     $ luarocks install PKG_NAME"
echo ""
echo "  + packages installed:"
echo "    - sundown   : " $sundown
echo "    - cwrap     : " $cwrap
echo "    - paths     : " $paths
echo "    - torch     : " $torch
echo "    - nn        : " $nn
echo "    - dok       : " $dok
echo "    - gnuplot   : " $gnuplot
[ -n "$cutorch" ] && echo "    - cutorch   : " $cutorch
[ -n "$cunn" ]    && echo "    - cunn      : " $cunn
#echo "    - qtlua     : " $qtlua
#echo "    - qttorch   : " $qttorch
echo "    - lfs       : " $lfs
echo "    - penlight  : " $penlight
echo "    - sys       : " $sys
echo "    - xlua      : " $xlua
echo "    - image     : " $image
echo "    - optim     : " $optim
echo "    - cjson     : " $cjson
echo "    - trepl     : " $trepl
echo ""
    
