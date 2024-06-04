# bsp_public

## Setup
```
docker pull daeseoklee/bsp_public:dev
```

## Run
```
docker run -it -p 6030:5000 --rm --name bsp_public daeseoklee/bsp_public:dev
bash scripts/example_inference.sh
```