# Dataset

## OSM Web Wizard
```powershell
conda activate thesis
python $env:SUMO_HOME/tools/osmWebWizard.py
```

## Area Size
### Original in Berlin
2.35 km Horizontal
1.55 km Vertical

### Original in Athens
3.05 km Horizontal
1.95 km Vertical

### Selected in Athens
2.35 km Horizontal
1.55 km Vertical

## Parameters
- Duration: 36000
- Add Polygons: True
- Car-only Network: True

## Vehicles
Disable all random traffic generation

## Vehicle Types
- [Vehicle Types](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#vehicle_types)
- [HBEFA3-based](https://sumo.dlr.de/docs/Models/Emissions/HBEFA3-based.html)
- [Vehicle Type Parameter Defaults](https://sumo.dlr.de/docs/Vehicle_Type_Parameter_Defaults.html)

## Fixed Trips
```
netedit .\dataset\athens-osmWebWizard\osm.net.xml.gz
duarouter -n .\dataset\athens-osmWebWizard\osm.net.xml.gz -r .\dataset\fixed-routes.rou.xml -o check-routes.rou.xml --ignore-errors
```

### 1. Omonoia to Evangelismos
- Starting Edge: 23182962 (Stadiou)
- Destination Edge: 169130585 (Leof. Vasilissis Sofias)

#### A. Via Akadimias
- 260124786#0 (Akadimias)
- 1209362820 (Pl. Filikis Eterias)
- 299645496 (Marasli)

#### B. Via Stadiou
- 299506410#0 (Stadiou)
- 221139568 (Leof. Vasilissis Amalias)
- -820421378#1 (Leof. Vasilissis Sofias)
