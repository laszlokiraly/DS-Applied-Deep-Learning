# Learnings

## dataset loading (validation of hrnet)

- if you use harddisk you should make sure that your process is the only reading from it, no other process should access the harddisk

- have multiple workers, e.g. 4 instead of only 1 (because 1 will only trash with high hd load)

- higher batch size -> more load on gpu -> more time for i/o -> optimal gpu utilization setting allows using slower harddisk (no need for ssd in this case)

best results:

```bash
ssd, workers 8, batch size 256, ~ 6.4 GB GPU Memory: 2:45 min
hd,  workers 4, batch size 128: ~ 3.0 GB GPU Memory:05 min
```

## dependencies

It turned out that for my usecase yacs (`pip install yacs`) and matplotlib (`conda install matplotlib`) sufficed. The hassle with shapely was probably not necessary at all. The pillow problem was not existing after clean installation of pytorch (`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`) in a new conda environment.
