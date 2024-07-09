docker build -t daeseoklee/bsp-inference:dep1 docker/dep1
if [ $? -ne 0 ]; then
    echo "Error building bsp-inference:dep1"
    exit 1
fi
docker push daeseoklee/bsp-inference:dep1

docker build -t daeseoklee/bsp-inference:dep2 docker/dep2
if [ $? -ne 0 ]; then
    echo "Error building bsp-inference:dep2"
    exit 1
fi
docker push daeseoklee/bsp-inference:dep2

docker build -t daeseoklee/bsp-inference . -f docker/Dockerfile
if [ $? -ne 0 ]; then
    echo "Error building bsp-inference"
    exit 1
fi
docker push daeseoklee/bsp-inference