netgenerate --grid --grid.number=6 --grid.length=300 --default.lanenumber=2 --default-junction-type traffic_light; randomTrips.py -e 7200 -p 0.75 --seed 42 --fringe-factor 5 --trip-attributes type=twin
newod: same, --seed 43 --fringe-factor 2
newodpeak: --seed 43 --fringe-factor 2 -p 0.65 (15% more trips)
