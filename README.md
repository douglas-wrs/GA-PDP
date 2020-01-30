# GA-DRT
A Multi-Objective Genetic Algorithm For Finding An Optimized Set Of Virtual Stops In A Collective Public Demand Responsive Transport Service

# ./osrm_walking

MacOS and Linux  

docker run -t -i -p 5001:5001 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5001 --algorithm mld /data/gyn-latest.osrm

docker run -t -i -p 5005:5005 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5005 --algorithm mld /data/gyn-latest.osrm

docker run -t -i -p 5003:5003 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5003 --algorithm mld /data/gyn-latest.osrm

Windows  

docker run -t -i -p 5001:5001 -v %cd%:/data osrm/osrm-backend osrm-routed --port 5001 --algorithm mld /data/gyn-latest.osrm

# ./osrm_driving

MacOS and Linux  

docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5000 --algorithm mld /data/gyn-latest.osrm

docker run -t -i -p 5004:5004 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5004 --algorithm mld /data/gyn-latest.osrm

docker run -t -i -p 5002:5002 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --port 5002 --algorithm mld /data/gyn-latest.osrm

Windows  

docker run -t -i -p 5000:5000 -v %cd%:/data osrm/osrm-backend osrm-routed --port 5000 --algorithm mld /data/gyn-latest.osrm
