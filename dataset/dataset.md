# Dataset

## OSM Web Wizard
For Windows:
```powershell
conda activate thesis
python $env:SUMO_HOME\tools\osmWebWizard.py
```

For Linux/MacOS:
```bash
conda activate thesis
python $SUMO_HOME/tools/osmWebWizard.py
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
```bash
netedit ./dataset/athens-osmWebWizard/osm.net.xml.gz
```

### Trip 1 - Omonoia to Evangelismos
- Starting Edge: 23182962 (Stadiou)
- Destination Edge: 169130585 (Leof. Vasilissis Sofias)

#### Route A - Via Akadimias
- 260124786#0 (Akadimias)
- 1209362820 (Pl. Filikis Eterias)
- 299645496 (Marasli)

#### Route B - Via Stadiou
- 299506410#0 (Stadiou)
- 221139568 (Leof. Vasilissis Amalias)
- -820421378#1 (Leof. Vasilissis Sofias)
